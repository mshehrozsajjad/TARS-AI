"""
module_body_state.py

Unified nervous system state layer for TARS-AI.

Aggregates data from emotions, drives, awareness, and sensors into a
single cached snapshot.  Produces a compact one-line prompt for the LLM
(replacing three separate verbose blocks) and applies bidirectional
coupling between drives and emotions on each interaction.

Architecture:
  Reflex layer  — IMU, startle, thermal (future, no LLM)
  State layer   — THIS MODULE: shared body state, updated cheaply
  Salience      — only notable changes bubble up
  LLM           — receives compact snapshot, not raw streams

The body state manager owns no data — it reads from existing modules
on demand and caches for a short TTL.  If it fails, the system degrades
gracefully to the original separate prompt blocks.
"""

import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from modules.module_messageQue import queue_message

# ── Singleton ────────────────────────────────────────────────────────────────

_instance = None


def get_body_state_manager():
    return _instance


# ── Bidirectional coupling constants ─────────────────────────────────────────

# Emotion → Drive adjustments (applied once per conversation turn)
# Keys are emotion axis names, values are {drive_name: delta}
_EMO_TO_DRIVE = {
    "joy":       {"boredom": -8, "social": -5},
    "sadness":   {"social": 5, "energy": -3},
    "curiosity": {"curiosity": -10, "boredom": -5},
    "love":      {"social": -10},
    "anger":     {"boredom": -3},
    "fear":      {"energy": -5},
}

# Drive → Emotion bias (applied when drive > threshold)
# Keys are drive names, values are {emotion_axis: boost_at_max}
_DRIVE_TO_EMO = {
    "social":    {"sadness": 0.08, "love": 0.04},    # lonely → tinge of sadness
    "boredom":   {"anger": 0.05},                      # bored → mild irritation
    "curiosity": {"curiosity": 0.06},                  # curious drive feeds curious mood
}
_DRIVE_COUPLING_THRESHOLD = 60  # drive must exceed this to influence emotions

# Mood modifier definitions (imported from prompt module at runtime)
_MOOD_ACTIVATION_THRESHOLD = 25


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BodyState:
    """Snapshot of TARS's complete internal + external state."""
    timestamp: float = 0.0

    # Emotions (8-axis radar, 0-100 each)
    emotions: dict = field(default_factory=dict)
    dominant_emotion: str = "neutral"

    # Drives (0-100 each)
    drives: dict = field(default_factory=dict)
    idle_minutes: float = 0.0

    # Awareness
    people_present: list = field(default_factory=list)
    scene: str = ""
    recent_events: list = field(default_factory=list)

    # Sensors (extensible)
    sensors: dict = field(default_factory=dict)

    # Derived
    time_of_day: str = "day"
    uptime_hours: float = 0.0
    off_duration: float = 0.0  # seconds TARS was powered off before this boot


# ── Manager ──────────────────────────────────────────────────────────────────

class BodyStateManager:
    """Central nervous system hub — aggregates all subsystem state."""

    CACHE_TTL = 2.0  # seconds — recompute at most every 2s

    def __init__(self, config, battery_module=None):
        global _instance
        _instance = self

        self._config = config
        self._battery = battery_module
        self._boot_time = time.time()

        self._cache: Optional[BodyState] = None
        self._cache_ts: float = 0.0
        self._lock = threading.Lock()

        # Extensible sensor registry: {name: callable}
        self._custom_sensors: dict = {}

        queue_message("LOAD: Body state system initialized")

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> BodyState:
        """Return a cached BodyState, recomputing if stale (>2s)."""
        now = time.monotonic()
        with self._lock:
            if self._cache and (now - self._cache_ts) < self.CACHE_TTL:
                return self._cache

        # Emotions
        emotions = {}
        dominant = "neutral"
        try:
            from modules.module_dashboard_data import get_emotional_state
            emotions = get_emotional_state()
            active = {k: v for k, v in emotions.items() if v > 0}
            if active:
                dominant = max(active, key=active.get)
        except Exception:
            pass

        # Drives
        drives = {}
        idle_minutes = 0.0
        off_duration = 0.0
        try:
            from modules.module_drives import get_drives_manager
            dm = get_drives_manager()
            if dm is not None:
                drives = dm.get_drives()
                off_duration = dm.get_off_duration()
                with dm._lock:
                    idle_minutes = (time.time() - dm._last_interaction) / 60.0
        except Exception:
            pass

        # Awareness
        people = []
        scene = ""
        events = []
        try:
            from modules.module_awareness import get_awareness_manager
            am = get_awareness_manager()
            if am is not None:
                people = am.get_present_people()
                scene = am.get_scene_description()
                events = am._presence.get_recent_events(minutes=10)
        except Exception:
            pass

        # Sensors
        sensors = self._read_sensors()

        # Time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            tod = "morning"
        elif 12 <= hour < 17:
            tod = "afternoon"
        elif 17 <= hour < 21:
            tod = "evening"
        else:
            tod = "night"

        state = BodyState(
            timestamp=time.time(),
            emotions=emotions,
            dominant_emotion=dominant,
            drives=drives,
            idle_minutes=idle_minutes,
            people_present=people,
            scene=scene,
            recent_events=events,
            sensors=sensors,
            time_of_day=tod,
            uptime_hours=(time.time() - self._boot_time) / 3600.0,
            off_duration=off_duration,
        )

        with self._lock:
            self._cache = state
            self._cache_ts = now

        return state

    # ── Compact prompt ───────────────────────────────────────────────────

    def get_compact_prompt(self) -> str:
        """Single compact line for LLM prompt injection.

        Replaces the three separate verbose blocks.
        Only includes notable state — neutral/default values are omitted.

        Example: [evening · Shehroz present 5m · mood: curious · wanting company · idle 15m]
        """
        s = self.snapshot()
        parts = []

        # Time of day
        parts.append(s.time_of_day)

        # Just woke up after being off (only include early in the session)
        if s.off_duration > 3600 and s.uptime_hours < 0.5:
            off_hours = s.off_duration / 3600
            if off_hours >= 24:
                parts.append(f"just powered on after {off_hours / 24:.0f} days off")
            else:
                parts.append(f"just powered on after {off_hours:.0f}h off")

        # People present
        if s.people_present:
            names = []
            for p in s.people_present:
                dur = p.get("duration_seconds", 0)
                name = p.get("name", "someone")
                if dur < 60:
                    names.append(f"{name} (just arrived)")
                elif dur < 3600:
                    names.append(f"{name} ({int(dur / 60)}m)")
                else:
                    names.append(f"{name} ({dur / 3600:.1f}h)")
            parts.append(", ".join(names) + " present")

        # Sensors (notable only)
        batt = s.sensors.get("battery_pct")
        if batt is not None:
            if batt < 50:
                charging = " charging" if s.sensors.get("charging") else ""
                parts.append(f"battery {batt}%{charging}")
        temp = s.sensors.get("cpu_temp_c")
        if temp is not None and temp > 65:
            parts.append(f"running warm ({temp:.0f}C)")

        # Mood (skip if neutral/low)
        if s.dominant_emotion != "neutral":
            intensity = s.emotions.get(s.dominant_emotion, 0)
            if intensity >= 15:
                parts.append(f"mood: {s.dominant_emotion}")

        # Drives (human-readable, only notable ones)
        _drive_labels = {
            "social":    ("wanting company", 50),
            "boredom":   ("restless", 50),
            "curiosity": ("curious", 60),
        }
        for drive, (label, threshold) in _drive_labels.items():
            val = s.drives.get(drive, 0)
            if val > threshold:
                parts.append(label)
        if s.drives.get("energy", 100) < 30:
            parts.append("fatigued")

        # Idle time (only if notable)
        if s.idle_minutes > 5:
            if s.idle_minutes >= 60:
                parts.append(f"idle {s.idle_minutes / 60:.0f}h")
            else:
                parts.append(f"idle {s.idle_minutes:.0f}m")

        # Scene (truncated)
        if s.scene:
            parts.append(f"scene: {s.scene[:50]}")

        # Recent events
        arrivals = [e["name"] for e in s.recent_events if e["type"] == "arrived"]
        departures = [e["name"] for e in s.recent_events if e["type"] == "departed"]
        if arrivals:
            # Deduplicate
            parts.append(f"just arrived: {', '.join(dict.fromkeys(arrivals))}")
        if departures:
            parts.append(f"recently left: {', '.join(dict.fromkeys(departures))}")

        if len(parts) <= 1:
            # Only time-of-day, nothing noteworthy
            return ""

        return "[" + " · ".join(parts) + "]"

    # ── Motion parameters (mood → speed / amplitude / fidget rate) ─────

    # Mood → motion mapping: (speed_factor, amplitude_factor, fidget_interval_s)
    _MOTION_PARAMS = {
        "joy":       (1.3, 1.2, 120),
        "curiosity": (1.0, 1.0, 180),
        "neutral":   (0.9, 0.8, 240),
        "sadness":   (0.5, 0.6, 360),
        "anger":     (1.2, 1.1, 150),
        "fear":      (0.6, 0.7, 300),
        "love":      (0.8, 0.9, 200),
        "surprise":  (1.1, 1.1, 150),
    }

    def get_motion_params(self) -> tuple:
        """Return (speed, amplitude, fidget_interval) based on current mood.

        speed: 0.3–1.5 — multiplier for servo speed_factor
        amplitude: 0.3–1.5 — scales gesture range of motion
        fidget_interval: seconds between idle fidgets
        """
        s = self.snapshot()

        # Low energy overrides mood — fatigue dominates motion
        energy = s.drives.get("energy", 100)
        if energy < 30:
            return (0.5, 0.4, 90)

        params = self._MOTION_PARAMS.get(s.dominant_emotion, (0.9, 0.8, 50))
        return params

    # ── Trait modifiers (unified emotion + drive) ────────────────────────

    def get_trait_modifiers(self) -> dict:
        """Combine emotion + drive modifiers into one dict.

        Called by _apply_mood_modifiers() in module_prompt.py.
        """
        mods = {}
        s = self.snapshot()

        # Emotion-based modifiers
        try:
            from modules.module_prompt import _MOOD_MODIFIERS
            for axis, intensity in s.emotions.items():
                if intensity < _MOOD_ACTIVATION_THRESHOLD:
                    continue
                axis_mods = _MOOD_MODIFIERS.get(axis)
                if not axis_mods:
                    continue
                scale = (intensity - _MOOD_ACTIVATION_THRESHOLD) / (100 - _MOOD_ACTIVATION_THRESHOLD)
                scale = 0.5 + 0.5 * scale  # 0.5 at threshold, 1.0 at max
                for trait, modifier in axis_mods.items():
                    mods[trait] = mods.get(trait, 0) + int(modifier * scale)
        except Exception:
            pass

        # Drive-based modifiers
        try:
            from modules.module_drives import get_drives_manager
            dm = get_drives_manager()
            if dm is not None:
                drive_mods = dm.get_drive_modifiers()
                for trait, modifier in drive_mods.items():
                    mods[trait] = mods.get(trait, 0) + modifier
        except Exception:
            pass

        return mods

    # ── Interaction routing + coupling ───────────────────────────────────

    def notify_interaction(self, user_text=None, reply_text=None,
                           emotion=None, axis_scores=None):
        """Route interaction to drives + apply bidirectional coupling.

        Called once per conversation turn from utterance_callback.
        """
        # 1. Forward to drives (existing behavior)
        dm = None
        try:
            from modules.module_drives import get_drives_manager
            dm = get_drives_manager()
            if dm is not None:
                dm.on_interaction(user_text, reply_text)
        except Exception:
            pass

        # 2. Emotion → Drive coupling
        if emotion and dm is not None:
            deltas = _EMO_TO_DRIVE.get(emotion, {})
            if deltas:
                try:
                    with dm._lock:
                        for drive_name, delta in deltas.items():
                            if drive_name in dm._drives:
                                dm._drives[drive_name] = max(0.0, min(100.0,
                                    dm._drives[drive_name] + delta))
                except Exception:
                    pass

        # 3. Drive → Emotion coupling (bias axis_scores before they're logged)
        if dm is not None and axis_scores is not None:
            try:
                drives = dm.get_drives()
                for drive_name, emo_map in _DRIVE_TO_EMO.items():
                    drive_val = drives.get(drive_name, 0)
                    if drive_val > _DRIVE_COUPLING_THRESHOLD:
                        scale = (drive_val - _DRIVE_COUPLING_THRESHOLD) / \
                                (100 - _DRIVE_COUPLING_THRESHOLD)
                        for emo_axis, strength in emo_map.items():
                            current = axis_scores.get(emo_axis, 0)
                            axis_scores[emo_axis] = min(1.0, current + strength * scale)
            except Exception:
                pass

        # 4. Invalidate cache so next snapshot() recomputes
        with self._lock:
            self._cache = None

    # ── Sensors ──────────────────────────────────────────────────────────

    def register_sensor(self, name: str, reader):
        """Register a custom sensor reader.

        Args:
            name: Unique sensor name (e.g. "imu", "touch")
            reader: Callable returning dict of {key: value} pairs.
        """
        self._custom_sensors[name] = reader

    def _read_sensors(self) -> dict:
        """Read all sensor values (built-in + registered)."""
        result = {}

        # Battery
        if self._battery is not None:
            try:
                result["battery_pct"] = self._battery.get_normalized_percentage()
                result["charging"] = getattr(self._battery, 'is_charging', lambda: False)()
            except Exception:
                pass

        # CPU temperature (read from sysfs on Pi)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                result["cpu_temp_c"] = int(f.read().strip()) / 1000.0
        except Exception:
            pass

        # Custom registered sensors
        for name, reader in self._custom_sensors.items():
            try:
                data = reader()
                if isinstance(data, dict):
                    result.update(data)
            except Exception:
                pass

        return result
