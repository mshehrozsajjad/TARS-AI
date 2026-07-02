"""
app_narrate.py

Standalone narration / voice-over mode for TARS.

Feed a script one line at a time from the terminal. After a short countdown,
TARS speaks the line aloud through the configured TTS backend while the DSI
screen UI shows the text. Lines may contain inline [gesture] tags to move the
body while speaking.

This is a self-contained tool — it does NOT start STT, the LLM, wake-word
detection, ChatUI, or idle body fidgets. It only loads config + TTS + UI +
servos, so it is safe to run for recording takes without TARS "coming alive"
on its own.

Usage:
    python app_narrate.py                       # defaults: 3s countdown, gestures on, UI on
    python app_narrate.py countdown=5           # 5 second countdown before each take
    python app_narrate.py gestures=off          # ignore [gesture] tags, speak only
    python app_narrate.py show_ui=false         # headless (no DSI screen output)

See NARRATE.md for the full option list, REPL commands, and gesture reference.
"""

# === Standard Libraries ===
import os
import sys
import re
import time
import asyncio
import threading
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# === Set up paths first (mirror app.py) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "modules"))
sys.path.append(os.getcwd())

# === Core Modules ===
from modules.module_config import load_config
from modules.module_messageQue import queue_message
from modules.module_state import set_tars_state, on_state_change, TarsState
from modules.module_tts import init_audio_output, play_audio_chunks, stop_tts_playback
# Import servoctl before gestures to fully resolve the servoctl <-> movements
# circular import (matches app.py's load order via module_main). Importing
# gestures first would load movements while servoctl is only partially
# initialized, raising "cannot import name 'step_forward'".
from modules.module_servoctl import initialize_servos
from modules.module_gestures import execute_gesture_async, GESTURE_NAMES

# === Load Configuration ===
CONFIG = load_config()
DEVICE_INFO = CONFIG.get("_device", {})
CHAR_NAME = CONFIG['CHAR'].get('character_name', 'TARS')
TTS_OPTION = CONFIG['TTS']['ttsoption']

# === Runtime options (overridable via CLI / REPL) ===
countdown_seconds = 3
gestures_enabled = True
show_ui = True

for arg in sys.argv[1:]:
    if "=" in arg:
        key, value = arg.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key == "countdown":
            try:
                countdown_seconds = max(0, int(value))
            except ValueError:
                pass
        elif key == "gestures":
            gestures_enabled = value in ("1", "true", "yes", "on")
        elif key == "show_ui":
            show_ui = value in ("1", "true", "yes", "on")


# === UI Manager (replicate app.py selection: lite on this Pi) ===
ui_manager = None
_use_lite_ui = False


def _init_ui(shutdown_event):
    """Bring up the DSI screen UI, matching app.py's capability-based choice."""
    global ui_manager, _use_lite_ui
    if not show_ui or not CONFIG["UI"].get("UI_enabled", False):
        return

    caps = DEVICE_INFO.get("capabilities")
    if caps is not None and not caps.can_use_ui:
        return

    _use_lite_ui = caps is not None and not caps.can_use_opengl
    try:
        if _use_lite_ui:
            from modules.module_ui_lite import UIManagerLite as UIManager
        else:
            from modules.module_ui import UIManager
        ui_manager = UIManager(
            shutdown_event=shutdown_event,
            battery_module=None,
            cpu_temp_module=None,
        )
        ui_manager.start()
        queue_message(f"LOAD: {'Lite' if _use_lite_ui else 'Full'} UI started for narration")
        set_tars_state(TarsState.BOOTING)
    except Exception as e:
        import traceback
        queue_message(f"WARNING: UI not available: {type(e).__name__}: {e}")
        traceback.print_exc()
        ui_manager = None


def _sync_state_to_ui(old_state, new_state):
    if ui_manager:
        ui_manager.set_tars_status(new_state.value)


# === Script line parsing ===
_TAG_RE = re.compile(r'\[([a-zA-Z_]+)\]')


def parse_line(line):
    """Split a script line into (spoken_text, [gesture, ...]).

    Tags matching a known gesture name are pulled out and returned as gestures
    (and removed from the spoken text). Any other [tag] is left untouched in the
    text so that TTS-native markup (e.g. ElevenLabs SSML <break>) passes through.
    """
    gestures = []

    def _replace(match):
        name = match.group(1).lower()
        if name in GESTURE_NAMES:
            gestures.append(name)
            return ""  # strip gesture tag from spoken text
        return match.group(0)  # keep unknown tag verbatim

    text = _TAG_RE.sub(_replace, line)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text, gestures


# === Speaking ===
def run_countdown(seconds):
    if seconds <= 0:
        return
    print()
    for remaining in range(seconds, 0, -1):
        print(f"  Recording in {remaining}...", flush=True)
        time.sleep(1)
    print("  >>> SPEAKING <<<\n", flush=True)


def speak_line(text, gestures):
    """Display the line on the UI, fire gestures, and speak it aloud (blocking)."""
    if ui_manager:
        ui_manager.deactivate_screensaver()
    set_tars_state(TarsState.TALKING)
    if ui_manager:
        # On the lite UI, TALKING is a no-op for status, so push the actual
        # text through update_data (char-name source renders in cyan).
        ui_manager.update_data(CHAR_NAME, text, CHAR_NAME)

    if gestures_enabled:
        for g in gestures:
            execute_gesture_async(g)

    try:
        asyncio.run(play_audio_chunks(text, TTS_OPTION))
    except Exception as e:
        queue_message(f"ERROR: TTS playback failed: {e}")
    finally:
        set_tars_state(TarsState.STANDBY)


# === REPL ===
HELP_TEXT = """
TARS Narration Mode — commands:
  <text>            Speak a line (after countdown). Use [gesture] tags inline.
  :replay           Re-speak the last line.
  :countdown N      Set countdown seconds (0 disables it).
  :gest on|off      Toggle gesture playback.
  :ui               Show current settings.
  :help             Show this help.
  :q / :quit        Exit narration mode.

Available gestures: {gestures}
""".strip()


def print_help():
    print(HELP_TEXT.format(gestures=", ".join(GESTURE_NAMES)))


def repl():
    global countdown_seconds, gestures_enabled
    last_text = None
    last_gestures = []

    print("\n" + "=" * 52)
    print("  TARS NARRATION MODE")
    print(f"  TTS backend : {TTS_OPTION}")
    print(f"  Countdown   : {countdown_seconds}s")
    print(f"  Gestures    : {'on' if gestures_enabled else 'off'}")
    print(f"  UI          : {'on' if ui_manager else 'off'}")
    print("  Type :help for commands, :q to quit.")
    print("=" * 52 + "\n")

    while True:
        try:
            raw = input("script> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting narration mode.")
            break

        line = raw.strip()
        if not line:
            continue

        # --- Commands ---
        if line in (":q", ":quit", ":exit"):
            print("Exiting narration mode.")
            break
        if line in (":help", ":h", ":?"):
            print_help()
            continue
        if line == ":ui":
            print(f"  countdown={countdown_seconds}s  gestures="
                  f"{'on' if gestures_enabled else 'off'}  "
                  f"ui={'on' if ui_manager else 'off'}  tts={TTS_OPTION}")
            continue
        if line == ":replay":
            if last_text is None:
                print("  Nothing to replay yet.")
                continue
            run_countdown(countdown_seconds)
            speak_line(last_text, last_gestures)
            continue
        if line.startswith(":countdown"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                countdown_seconds = int(parts[1])
                print(f"  Countdown set to {countdown_seconds}s.")
            else:
                print("  Usage: :countdown N")
            continue
        if line.startswith(":gest"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                gestures_enabled = parts[1].lower() == "on"
                print(f"  Gestures {'enabled' if gestures_enabled else 'disabled'}.")
            else:
                print("  Usage: :gest on|off")
            continue
        if line.startswith(":"):
            print(f"  Unknown command: {line}  (try :help)")
            continue

        # --- Script line ---
        text, gestures = parse_line(line)
        if not text:
            print("  (line had no speakable text after removing tags)")
            continue
        if gestures:
            print(f"  gestures: {', '.join(gestures)}")
        run_countdown(countdown_seconds)
        last_text, last_gestures = text, gestures
        speak_line(text, gestures)


# === Main ===
def main():
    shutdown_event = threading.Event()

    _init_ui(shutdown_event)
    on_state_change(_sync_state_to_ui)

    init_audio_output()

    if gestures_enabled:
        try:
            initialize_servos()
            queue_message("LOAD: Servos initialized for narration gestures")
        except Exception as e:
            queue_message(f"WARNING: Servo init failed, gestures disabled: {e}")

    set_tars_state(TarsState.STANDBY)

    try:
        repl()
    finally:
        stop_tts_playback()
        set_tars_state(TarsState.STANDBY)
        shutdown_event.set()
        if ui_manager:
            try:
                ui_manager.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
