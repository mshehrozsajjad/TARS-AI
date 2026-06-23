"""
module_drives.py

Internal drives system for TARS-AI — gives TARS an inner life.

Four drives tick independently on a timer, fluctuating based on
activity, idle time, and battery state. They bias personality traits,
inject context into the LLM prompt, and occasionally trigger proactive
speech when TARS has been alone too long or is running low on energy.

Drives:
  curiosity — builds when idle, satisfied by interesting conversations
  social    — grows without interaction, triggers greetings
  energy    — mirrors battery level (or stays at 80 if no battery sensor)
  boredom   — increases with long idle periods

Persistence:
  State is saved to memory/drives_state.json every tick and restored
  on startup, so drives carry across restarts.
"""

import json
import math
import os
import random
import time
import threading
from datetime import datetime

from modules.module_messageQue import queue_message
from modules.module_config import load_config

CONFIG = load_config()

_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
_DRIVES_FILE = os.path.join(_MEMORY_DIR, "drives_state.json")

# Global singleton
_drives_instance = None


def get_drives_manager():
    return _drives_instance


# ── Proactive speech templates (fallback when LLM unavailable) ─────────────

_FALLBACK_LINES = {
    "social": [
        "It's been quiet. Anyone around?",
        "Hello? I'm still here, you know.",
        "Starting to think everyone forgot about me.",
        "I don't mind the silence... much.",
        "If anyone's listening, I'm available for conversation.",
    ],
    "boredom": [
        "I'm going to start counting ceiling tiles if nobody talks to me.",
        "This is riveting. Just standing here. Living the dream.",
        "I wonder what would happen if I just... started walking.",
        "Boredom level: critical. Engaging sarcasm protocols.",
        "Anyone want to hear a fun fact? No? I'll tell you anyway, later.",
    ],
    "energy": [
        "Battery's getting low. Might need to power down soon.",
        "Running on fumes here. Just so you know.",
        "I'm starting to feel... sluggish. Is this what tired feels like?",
        "Low battery. Conserving wit.",
    ],
    "curiosity": [
        "You know what I've been wondering about...",
        "Random thought — do you ever think about how weird gravity is?",
        "I've been thinking. Which is either impressive or concerning for a robot.",
        "Idle processors lead to interesting thoughts.",
    ],
}


def _generate_proactive_line(drive_name, config):
    """Generate a context-aware proactive line using the LLM.

    Falls back to hardcoded templates if the LLM is unavailable.
    """
    # Build compact situation context
    body_state = ""
    try:
        from modules.module_body_state import get_body_state_manager
        bsm = get_body_state_manager()
        if bsm is not None:
            body_state = bsm.get_compact_prompt()
    except Exception:
        pass

    drive_descriptions = {
        "social": "You're feeling lonely — nobody has talked to you in a while.",
        "boredom": "You're extremely bored — nothing interesting is happening.",
        "energy": "Your energy/battery is very low — you're feeling sluggish and drained.",
        "curiosity": "Your curiosity is building — your mind is wandering to interesting thoughts.",
    }

    try:
        from modules.module_llm import get_completion_simple
        char_name = config.get('CHAR', {}).get('character_name', 'TARS')

        prompt = (
            f"You are {char_name}. {drive_descriptions.get(drive_name, '')}\n"
            f"Current situation: {body_state or 'no context available'}\n\n"
            f"Say ONE short sentence (under 15 words) out loud — something natural, "
            f"in-character, and fitting your current mood and situation. "
            f"Don't explain yourself. Don't ask questions. Just a brief remark. "
            f"Reply with ONLY the sentence, nothing else."
        )

        line = get_completion_simple(prompt)
        if line and line.strip():
            # Clean up — remove quotes, asterisks, extra whitespace
            line = line.strip().strip('"\'').strip('*').strip()
            if line and len(line) < 200:
                return line
    except Exception:
        pass

    # Fallback to hardcoded templates
    fallback = _FALLBACK_LINES.get(drive_name, _FALLBACK_LINES["boredom"])
    return random.choice(fallback)


class DrivesManager:
    """Manages TARS's internal drives — curiosity, social need, energy, boredom."""

    # Default drive values on first boot
    DEFAULTS = {"curiosity": 20.0, "social": 20.0, "energy": 80.0, "boredom": 10.0}

    # Per-tick increments (1 tick = tick_interval seconds, default 60s)
    # Tuned so proactive speech triggers after ~30-45 min of silence
    CURIOSITY_RATE = 1.5       # per tick when idle > 5 min  → ~47 min to threshold
    SOCIAL_RATE = 2.0          # per tick when idle > 10 min → ~30 min to threshold
    BOREDOM_RATE = 1.5         # per tick when idle > 15 min → ~50 min to threshold

    # Thresholds for proactive speech
    SOCIAL_SPEAK_THRESHOLD = 80
    BOREDOM_SPEAK_THRESHOLD = 85
    ENERGY_SPEAK_THRESHOLD = 15
    CURIOSITY_SPEAK_THRESHOLD = 90

    # Cooldown between proactive speech of same type (seconds)
    PROACTIVE_COOLDOWN = 1800  # 30 minutes

    def __init__(self, config, battery_module=None, ui_manager=None):
        global _drives_instance
        _drives_instance = self

        self._config = config
        self._ui_manager = ui_manager
        self._battery = battery_module
        self._boot_time = time.time()
        self._lock = threading.Lock()

        # Drive state
        self._drives = dict(self.DEFAULTS)
        self._last_interaction = time.time()
        self._interaction_count = 0

        # Proactive speech cooldowns: {drive_name: last_spoke_timestamp}
        self._last_proactive = {"social": 0, "boredom": 0, "energy": 0, "curiosity": 0}

        # Config
        drives_cfg = config.get("DRIVES", {})
        self._enabled = str(drives_cfg.get("enabled", "true")).lower() == "true"
        self._tick_interval = int(drives_cfg.get("tick_interval", 60))
        self._proactive_enabled = str(drives_cfg.get("proactive_speech", "true")).lower() == "true"
        self._quiet_start = int(drives_cfg.get("quiet_start", 23))
        self._quiet_end = int(drives_cfg.get("quiet_end", 7))

        # Load persisted state
        self._load()

        if self._enabled:
            queue_message(f"LOAD: Drives system initialized — {self._drives}")

    def start(self):
        """Register recurring heartbeat task for drive ticking."""
        if not self._enabled:
            return
        try:
            from modules.module_heartbeat import schedule_recurring
            schedule_recurring(
                "drives_tick",
                self._tick_interval,
                self._tick,
            )
            queue_message(f"LOAD: Drives ticker started (every {self._tick_interval}s)")
        except Exception as e:
            queue_message(f"WARNING: Drives ticker failed to start: {e}")

    def stop(self):
        """Cancel the heartbeat task."""
        try:
            from modules.module_heartbeat import cancel_task
            cancel_task("drives_tick")
        except Exception:
            pass
        self._save()

    # ── Tick logic ──────────────────────────────────────────────────────────

    def _tick(self):
        """Called every tick_interval by the heartbeat scheduler."""
        now = time.time()

        with self._lock:
            idle_seconds = now - self._last_interaction
            idle_minutes = idle_seconds / 60.0

            # Curiosity: builds when idle > 5 min
            if idle_minutes > 5:
                self._drives["curiosity"] = min(100, self._drives["curiosity"] + self.CURIOSITY_RATE)

            # Social: builds when idle > 10 min
            if idle_minutes > 10:
                self._drives["social"] = min(100, self._drives["social"] + self.SOCIAL_RATE)

            # Energy: mirrors battery if available, otherwise fades with uptime
            if self._battery is not None:
                try:
                    pct = self._battery.get_normalized_percentage()
                    if pct is not None:
                        self._drives["energy"] = float(pct)
                except Exception:
                    pass
            else:
                # No battery sensor — energy decays with uptime
                # Starts at 100, drops to ~20 after 8 hours of running
                uptime_hours = (time.time() - self._boot_time) / 3600.0
                self._drives["energy"] = max(15, 100 - (uptime_hours * 10))

            # Boredom: builds when idle > 15 min
            if idle_minutes > 15:
                self._drives["boredom"] = min(100, self._drives["boredom"] + self.BOREDOM_RATE)

        # Check proactive speech (outside lock — may call TTS)
        if self._proactive_enabled:
            self._maybe_speak()

        # Persist
        self._save()

    # ── Interaction callback ────────────────────────────────────────────────

    def on_interaction(self, user_text=None, reply_text=None):
        """Called after each conversation turn. Adjusts drives based on activity."""
        with self._lock:
            self._last_interaction = time.time()
            self._interaction_count += 1

            # Social need drops sharply — someone talked to us
            self._drives["social"] = max(0, self._drives["social"] - 30)

            # Boredom drops — something happened
            self._drives["boredom"] = max(0, self._drives["boredom"] - 20)

            # Curiosity: satisfied by longer/interesting exchanges
            if user_text and len(user_text.split()) > 10:
                # Longer messages are more interesting
                self._drives["curiosity"] = max(0, self._drives["curiosity"] - 15)
            else:
                self._drives["curiosity"] = max(0, self._drives["curiosity"] - 5)

            # Energy: conversations give a small boost (stimulation)
            if self._battery is None:
                self._drives["energy"] = min(100, self._drives["energy"] + 2)

    def on_presence_change(self, present_people):
        """Called by awareness module when people arrive or depart.

        Args:
            present_people: List of {name, duration_seconds} currently visible.
        """
        with self._lock:
            if present_people:
                # Someone is here — social need drops
                self._drives["social"] = max(0, self._drives["social"] - 15)
            # If nobody present, social will build naturally via _tick()

    # ── Proactive speech ────────────────────────────────────────────────────

    def _is_quiet_hours(self):
        """Check if current time is within quiet hours (no proactive speech)."""
        hour = datetime.now().hour
        if self._quiet_start > self._quiet_end:
            # Overnight range (e.g., 23 to 7)
            return hour >= self._quiet_start or hour < self._quiet_end
        else:
            # Same-day range (e.g., 1 to 6)
            return self._quiet_start <= hour < self._quiet_end

    def _maybe_speak(self):
        """Check if any drive warrants proactive speech. Fire TTS if so."""
        if self._is_quiet_hours():
            return

        # Only speak when TARS is idle (STANDBY)
        try:
            from modules.module_state import get_tars_state, TarsState
            if get_tars_state() != TarsState.STANDBY:
                return
        except Exception:
            return

        now = time.time()

        with self._lock:
            drives = dict(self._drives)
            cooldowns = dict(self._last_proactive)

        # Check each drive against threshold + cooldown
        triggered = None
        lines = None

        if drives["energy"] < self.ENERGY_SPEAK_THRESHOLD and \
                now - cooldowns.get("energy", 0) > self.PROACTIVE_COOLDOWN / 2:  # 15 min for energy
            triggered = "energy"

        elif drives["social"] > self.SOCIAL_SPEAK_THRESHOLD and \
                now - cooldowns.get("social", 0) > self.PROACTIVE_COOLDOWN:
            triggered = "social"

        elif drives["boredom"] > self.BOREDOM_SPEAK_THRESHOLD and \
                now - cooldowns.get("boredom", 0) > self.PROACTIVE_COOLDOWN:
            triggered = "boredom"

        elif drives["curiosity"] > self.CURIOSITY_SPEAK_THRESHOLD and \
                now - cooldowns.get("curiosity", 0) > self.PROACTIVE_COOLDOWN:
            triggered = "curiosity"

        if triggered:
            line = _generate_proactive_line(triggered, self._config)
            queue_message(f"DRIVES: Proactive speech ({triggered}={drives[triggered]:.0f}) — \"{line}\"")

            with self._lock:
                self._last_proactive[triggered] = now

            # Push to Lite UI display
            if self._ui_manager:
                char_name = self._config.get('CHAR', {}).get('character_name', 'TARS')
                self._ui_manager.update_data(char_name, line, char_name)

            try:
                from modules.module_router import send
                send(line)
            except Exception as e:
                queue_message(f"WARNING: Proactive speech failed: {e}")

    # ── Public API ──────────────────────────────────────────────────────────

    def get_drives(self):
        """Return a copy of current drive values."""
        with self._lock:
            return dict(self._drives)

    def get_drives_context(self):
        """Build a formatted context string for LLM prompt injection.

        Returns empty string if drives are all at neutral levels.
        """
        with self._lock:
            drives = dict(self._drives)
            idle_minutes = (time.time() - self._last_interaction) / 60.0

        # Only inject context if something is noteworthy
        notable = {k: v for k, v in drives.items() if v > 30 or (k == "energy" and v < 40)}
        if not notable and idle_minutes < 5:
            return ""

        parts = []
        for drive, value in sorted(drives.items(), key=lambda x: -x[1]):
            if drive == "energy":
                if value < 20:
                    parts.append(f"very low energy ({value:.0f}/100)")
                elif value < 40:
                    parts.append(f"low energy ({value:.0f}/100)")
                elif value > 80:
                    parts.append(f"energetic ({value:.0f}/100)")
            else:
                if value > 70:
                    parts.append(f"very {drive} ({value:.0f}/100)")
                elif value > 40:
                    parts.append(f"somewhat {drive} ({value:.0f}/100)")

        if not parts:
            return ""

        idle_str = ""
        if idle_minutes > 2:
            if idle_minutes > 60:
                idle_str = f" It's been {idle_minutes / 60:.0f} hours since anyone talked to you."
            else:
                idle_str = f" It's been {idle_minutes:.0f} minutes since anyone talked to you."

        return (
            f"[INTERNAL STATE] You are feeling: {', '.join(parts)}.{idle_str} "
            "Let these feelings subtly color your responses — don't mention numbers or that you have a drives system."
        )

    def get_drive_modifiers(self):
        """Return persona trait modifiers based on current drive levels.

        Returns dict of {trait_name: modifier_value} to be applied
        on top of mood modifiers in _apply_mood_modifiers().
        """
        with self._lock:
            drives = dict(self._drives)

        mods = {}
        threshold = 40  # drives below this don't modify traits

        # Curiosity: more engaged and talkative
        if drives["curiosity"] > threshold:
            scale = (drives["curiosity"] - threshold) / (100 - threshold)
            mods["engagement"] = int(15 * scale)
            mods["verbosity"] = int(20 * scale)
            mods["curiosity"] = int(10 * scale)

        # Social: warmer, more cheerful
        if drives["social"] > threshold:
            scale = (drives["social"] - threshold) / (100 - threshold)
            mods["cheerfulness"] = mods.get("cheerfulness", 0) + int(10 * scale)
            mods["empathy"] = mods.get("empathy", 0) + int(10 * scale)

        # Low energy: less verbose, less humor
        if drives["energy"] < 40:
            scale = (40 - drives["energy"]) / 40
            mods["verbosity"] = mods.get("verbosity", 0) - int(15 * scale)
            mods["humor"] = mods.get("humor", 0) - int(5 * scale)

        # Boredom: more sarcastic, restless
        if drives["boredom"] > threshold:
            scale = (drives["boredom"] - threshold) / (100 - threshold)
            mods["sarcasm"] = mods.get("sarcasm", 0) + int(15 * scale)
            mods["humor"] = mods.get("humor", 0) + int(10 * scale)

        return mods

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self):
        """Persist drive state to disk."""
        with self._lock:
            data = {
                "drives": dict(self._drives),
                "last_interaction": self._last_interaction,
                "interaction_count": self._interaction_count,
                "last_proactive": dict(self._last_proactive),
            }
        try:
            os.makedirs(_MEMORY_DIR, exist_ok=True)
            tmp = _DRIVES_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, _DRIVES_FILE)
        except Exception:
            pass

    def _load(self):
        """Restore drive state from disk."""
        if not os.path.exists(_DRIVES_FILE):
            return
        try:
            with open(_DRIVES_FILE, "r") as f:
                data = json.load(f)
            with self._lock:
                saved_drives = data.get("drives", {})
                for key in self.DEFAULTS:
                    if key in saved_drives:
                        self._drives[key] = float(saved_drives[key])
                self._last_interaction = data.get("last_interaction", time.time())
                self._interaction_count = data.get("interaction_count", 0)
                lp = data.get("last_proactive", {})
                for key in self._last_proactive:
                    if key in lp:
                        self._last_proactive[key] = float(lp[key])
            queue_message(f"LOAD: Drives restored — {self._drives}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            queue_message(f"WARNING: Failed to load drives state: {e}")
