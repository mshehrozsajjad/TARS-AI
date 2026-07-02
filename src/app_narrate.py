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
    python app_narrate.py port=5555             # take commands over the network
                                                #   (connect with: nc <pi-ip> 5555)
    python app_narrate.py model=inherit         # use config.ini's ElevenLabs model
                                                #   (default is eleven_v3 for emotion tags)

See NARRATE.md for the full option list, REPL commands, and gesture reference.
"""

# === Standard Libraries ===
import os
import sys
import re
import time
import signal
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
control_port = 0          # 0 = local stdin REPL; >0 = TCP control server
control_host = "0.0.0.0"  # bind address when control_port is set
# Narration defaults to ElevenLabs v3 so inline emotion audio tags ([excited],
# [whispers], [sarcastic], ...) work. Pass model=inherit to use config.ini's
# model instead, or model=<id> to force a specific one. Only applied when the
# active TTS backend is elevenlabs.
tts_model = "eleven_v3"

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
        elif key == "port":
            try:
                control_port = max(0, int(value))
            except ValueError:
                pass
        elif key == "host":
            control_host = value
        elif key == "model":
            tts_model = value

# === Apply TTS model override for this (standalone) narration process only ===
# load_config() returns a cached singleton shared with module_elevenlabs, so
# setting it here redirects narration's synthesis. This process is separate from
# the main TARS app, so the running robot's configured model is untouched.
if TTS_OPTION == "elevenlabs" and tts_model and tts_model != "inherit":
    CONFIG['TTS']['elevenlabs_model'] = tts_model

# === Shared state for the last spoken take (used by :replay) ===
last_text = None
last_gestures = []


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
def run_countdown(seconds, emit):
    if seconds <= 0:
        return
    emit("")
    for remaining in range(seconds, 0, -1):
        emit(f"  Recording in {remaining}...")
        time.sleep(1)
    emit("  >>> SPEAKING <<<")
    emit("")


def speak_line(text, gestures, emit):
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


# === Command handling (shared by local REPL and TCP control server) ===
HELP_TEXT = """
TARS Narration Mode — commands:
  <text>            Speak a line (after countdown). Use [gesture] tags inline.
  :replay           Re-speak the last line.
  :countdown N      Set countdown seconds (0 disables it).
  :gest on|off      Toggle gesture playback.
  :ui               Show current settings.
  :help             Show this help.
  :q / :quit        Exit narration mode (stops the app + display).

Available gestures: {gestures}
""".strip()


def _stdout_emit(msg=""):
    print(msg, flush=True)


def banner(emit):
    emit("")
    emit("=" * 52)
    emit("  TARS NARRATION MODE")
    emit(f"  TTS backend : {TTS_OPTION}")
    if TTS_OPTION == "elevenlabs":
        emit(f"  TTS model   : {CONFIG['TTS'].get('elevenlabs_model', '?')}")
    emit(f"  Countdown   : {countdown_seconds}s")
    emit(f"  Gestures    : {'on' if gestures_enabled else 'off'}")
    emit(f"  UI          : {'on' if ui_manager else 'off'}")
    emit("  Type :help for commands, :q to quit.")
    emit("=" * 52)
    emit("")


def process_command(raw, emit):
    """Handle one input line. Returns 'quit' to end the session, else 'continue'."""
    global countdown_seconds, gestures_enabled, last_text, last_gestures

    line = raw.strip()
    if not line:
        return "continue"

    # --- Commands ---
    if line in (":q", ":quit", ":exit"):
        emit("Exiting narration mode.")
        return "quit"
    if line in (":help", ":h", ":?"):
        emit(HELP_TEXT.format(gestures=", ".join(GESTURE_NAMES)))
        return "continue"
    if line == ":ui":
        emit(f"  countdown={countdown_seconds}s  gestures="
             f"{'on' if gestures_enabled else 'off'}  "
             f"ui={'on' if ui_manager else 'off'}  tts={TTS_OPTION}")
        return "continue"
    if line == ":replay":
        if last_text is None:
            emit("  Nothing to replay yet.")
            return "continue"
        run_countdown(countdown_seconds, emit)
        speak_line(last_text, last_gestures, emit)
        return "continue"
    if line.startswith(":countdown"):
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            countdown_seconds = int(parts[1])
            emit(f"  Countdown set to {countdown_seconds}s.")
        else:
            emit("  Usage: :countdown N")
        return "continue"
    if line.startswith(":gest"):
        parts = line.split()
        if len(parts) == 2 and parts[1].lower() in ("on", "off"):
            gestures_enabled = parts[1].lower() == "on"
            emit(f"  Gestures {'enabled' if gestures_enabled else 'disabled'}.")
        else:
            emit("  Usage: :gest on|off")
        return "continue"
    if line.startswith(":"):
        emit(f"  Unknown command: {line}  (try :help)")
        return "continue"

    # --- Script line ---
    text, gestures = parse_line(line)
    if not text:
        emit("  (line had no speakable text after removing tags)")
        return "continue"
    if gestures:
        emit(f"  gestures: {', '.join(gestures)}")
    run_countdown(countdown_seconds, emit)
    last_text, last_gestures = text, gestures
    speak_line(text, gestures, emit)
    return "continue"


def repl_stdin():
    """Local terminal REPL (used when no control port is set)."""
    banner(_stdout_emit)
    while True:
        try:
            raw = input("script> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting narration mode.")
            break
        if process_command(raw, _stdout_emit) == "quit":
            break


def serve_control(host, port, shutdown_event):
    """TCP control server so commands can be sent from another machine/terminal.

    One client is served at a time — the robot is a single physical device, so
    takes must be serialized anyway. Connect from anywhere on the LAN with:
        nc <pi-ip> <port>
    A local status line is still printed to the launching terminal via
    queue_message, but no input is read from it.
    """
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    queue_message(f"LOAD: Narration control server listening on {host}:{port}")
    print(f"\nRemote control ready. From another terminal run:\n"
          f"    nc <pi-ip> {port}\n", flush=True)

    while not shutdown_event.is_set():
        try:
            conn, addr = srv.accept()
        except OSError:
            break
        queue_message(f"INFO: Narration client connected from {addr[0]}")
        rfile = conn.makefile("r", encoding="utf-8")
        wfile = conn.makefile("w", encoding="utf-8")

        def emit(msg=""):
            try:
                wfile.write(msg + "\n")
                wfile.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

        quit_requested = False
        try:
            banner(emit)
            emit("script> ")
            for raw in rfile:
                if process_command(raw, emit) == "quit":
                    quit_requested = True
                    break
                emit("script> ")
        except (ConnectionError, OSError):
            pass
        finally:
            for f in (rfile, wfile):
                try:
                    f.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass
            queue_message("INFO: Narration client disconnected")

        if quit_requested:
            break

    srv.close()


# === Shutdown / cleanup ===
_shutdown_event = threading.Event()
_cleaned_up = threading.Event()


def _cleanup():
    """Release audio, TTS and the display cleanly. Idempotent."""
    if _cleaned_up.is_set():
        return
    _cleaned_up.set()
    try:
        stop_tts_playback()
    except Exception:
        pass
    try:
        set_tars_state(TarsState.STANDBY)
    except Exception:
        pass
    _shutdown_event.set()
    if ui_manager:
        try:
            ui_manager.stop()
            # Give the UI thread a moment to hit pygame.quit() so the DSI/DRM
            # display is released — otherwise NoMachine can't grab it.
            ui_manager.join(timeout=2)
        except Exception:
            pass


def _handle_signal(signum, frame):
    """SIGTERM/SIGINT -> clean shutdown so `pkill` releases the display.

    Without this, the default SIGTERM action kills the process without running
    pygame.quit(), which can leave the screen grabbed/frozen for NoMachine.
    """
    name = signal.Signals(signum).name
    queue_message(f"INFO: Narration received {name}, shutting down.")
    _cleanup()
    os._exit(0)


# === Main ===
def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _init_ui(_shutdown_event)
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
        if control_port > 0:
            serve_control(control_host, control_port, _shutdown_event)
        else:
            repl_stdin()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
