"""
Module: LiveKit Client

Lightweight LiveKit RTC client that runs on the Pi as a room participant.
No AI processing — the agent runs on LiveKit Cloud.

Responsibilities:
  - Publish Pi's microphone as an audio track
  - Receive agent's audio track → play on Pi's physical speaker
  - Receive agent's video track → render on Pi's Pygame display
  - Expose physical actions (servos, movement, emotion) as RPC handlers
    that the cloud agent can invoke via tool calls

Uses the `livekit` RTC SDK (not `livekit-agents`).
Requires: pip install livekit
"""

import os
import json
import asyncio
import logging
import threading
import time
import numpy as np
from modules.module_config import load_config
from modules.module_messageQue import queue_message
from modules.module_state import set_tars_state, TarsState

# Suppress non-fatal QueueFull warnings from livekit's internal audio mixer
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

CONFIG = load_config()


# ── Pygame video display (lightweight, low-res) ─────────────────────

class PygameDisplay:
    """Low-res Pygame video display for Pi.

    Receives RGBA frames, downscales via numpy stride skip,
    rotates the tiny surface, then scales up to fill screen.
    Target: 25fps with minimal CPU.
    """

    TARGET_FPS = 25
    # Downscale factor — take every Nth pixel in each dimension
    DOWNSAMPLE = 3

    def __init__(self):
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop,
            name="PygameDisplay",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def set_frame(self, rgba_array):
        """Accept RGBA numpy array from video handler."""
        with self._frame_lock:
            self._frame = rgba_array

    def _render_loop(self):
        # Prevent SDL from touching audio devices
        old_drv = os.environ.get("SDL_AUDIODRIVER")
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        import pygame
        pygame.display.init()

        if old_drv is None:
            os.environ.pop("SDL_AUDIODRIVER", None)
        else:
            os.environ["SDL_AUDIODRIVER"] = old_drv

        # Auto-detect actual screen
        info = pygame.display.Info()
        hw_w = info.current_w   # 800
        hw_h = info.current_h   # 480

        screen = pygame.display.set_mode(
            (hw_w, hw_h), pygame.FULLSCREEN | pygame.NOFRAME
        )
        pygame.mouse.set_visible(False)
        screen.fill((0, 0, 0))
        pygame.display.flip()

        # Screen is landscape (800x480) but mounted portrait on TARS
        # Logical view is portrait: 480 x 800
        logical_w = hw_h   # 480
        logical_h = hw_w   # 800

        queue_message(f"LIVEKIT: Pygame display ready hw={hw_w}x{hw_h} "
                      f"logical={logical_w}x{logical_h} @ {self.TARGET_FPS}fps")

        interval = 1.0 / self.TARGET_FPS
        last_frame = None

        while self._running:
            t0 = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            with self._frame_lock:
                new = self._frame
                self._frame = None

            if new is not None:
                last_frame = new

            if last_frame is not None:
                try:
                    # Downscale with stride skip — very fast, no interpolation
                    ds = self.DOWNSAMPLE
                    small = last_frame[::ds, ::ds, :3]
                    small = np.ascontiguousarray(small)
                    sh, sw = small.shape[:2]

                    # Create tiny surface
                    surface = pygame.image.frombuffer(
                        small.data, (sw, sh), "RGB"
                    )

                    # Scale tiny surface to fill portrait logical view
                    scale = max(logical_w / sw, logical_h / sh)
                    scaled = pygame.transform.scale(
                        surface, (int(sw * scale), int(sh * scale))
                    )

                    # Center crop to logical size
                    cx = (scaled.get_width() - logical_w) // 2
                    cy = (scaled.get_height() - logical_h) // 2
                    logical = pygame.Surface((logical_w, logical_h))
                    logical.blit(scaled, (-cx, -cy))

                    # Rotate 270° to match physical screen
                    rotated = pygame.transform.rotate(logical, 270)

                    screen.blit(rotated, (0, 0))
                    pygame.display.flip()
                except Exception:
                    pass

            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        pygame.quit()
        queue_message("LIVEKIT: Pygame display stopped")


_pygame_display = None


def _start_pygame_display():
    global _pygame_display
    if _pygame_display is None:
        _pygame_display = PygameDisplay()
    if not _pygame_display._running:
        _pygame_display.start()
    return _pygame_display


def _stop_pygame_display():
    global _pygame_display
    if _pygame_display is not None:
        _pygame_display.stop()
        _pygame_display = None


# ── Browser-based video display ──────────────────────────────────────

_display_server_started = False
_display_room_name = None
_display_livekit_url = None
_browser_process = None


def _start_display_server(livekit_url, room_name, port=8888):
    """Start a tiny HTTP server that serves the video display page
    and a /livekit-token endpoint. Launches Chromium in kiosk mode."""
    global _display_server_started, _display_room_name, _display_livekit_url
    global _browser_process

    if _display_server_started:
        # Update room name for new sessions
        _display_room_name = room_name
        return

    _display_livekit_url = livekit_url
    _display_room_name = room_name

    from http.server import HTTPServer, BaseHTTPRequestHandler
    import subprocess

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_dir, "www", "templates", "livekit_display.html")

    with open(template_path, "r") as f:
        html_content = f.read()

    class DisplayHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/livekit-token":
                # Generate a view-only token for the browser
                token = _generate_token(
                    _display_room_name, "tars-display"
                )
                lk_cfg = CONFIG["LIVEKIT"]
                body = json.dumps({
                    "url": _display_livekit_url,
                    "token": token,
                    "browser_audio": not lk_cfg["play_local_audio"],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/", "/livekit-display"):
                body = html_content.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/close":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"closing")
                threading.Thread(target=_stop_display, daemon=True).start()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # silence request logs

    server = HTTPServer(("127.0.0.1", port), DisplayHandler)

    def _serve():
        queue_message(f"LIVEKIT: Display server on http://127.0.0.1:{port}")
        server.serve_forever()

    t = threading.Thread(target=_serve, name="LiveKitDisplayServer", daemon=True)
    t.start()
    _display_server_started = True

    # Give the server a moment to bind
    time.sleep(0.5)

    # Launch Chromium in kiosk mode — fullscreen, no UI chrome
    # Binary name varies: "chromium-browser" (older Pi OS) vs "chromium" (newer)
    import shutil
    chromium_bin = shutil.which("chromium-browser") or shutil.which("chromium")
    if not chromium_bin:
        queue_message("WARNING: chromium not found — video display unavailable")
        return

    try:
        _browser_process = subprocess.Popen(
            [
                chromium_bin,
                "--kiosk",
                "--noerrdialogs",
                "--disable-infobars",
                "--disable-session-crashed-bubble",
                "--autoplay-policy=no-user-gesture-required",
                "--check-for-update-interval=31536000",
                "--disable-features=TranslateUI",
                "--no-first-run",
                "--password-store=basic",
                f"http://127.0.0.1:{port}/livekit-display",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        queue_message("LIVEKIT: Chromium kiosk launched for video display")
    except Exception as e:
        queue_message(f"WARNING: Could not launch Chromium — {e}")


def _stop_display():
    """Kill the browser process on shutdown."""
    global _browser_process
    if _browser_process is not None:
        try:
            _browser_process.terminate()
            _browser_process.wait(timeout=3)
        except Exception:
            try:
                _browser_process.kill()
            except Exception:
                pass
        _browser_process = None
    _stop_pygame_display()


# ── Wake word gate (hybrid: local detection + mic mute/unmute) ──────

class WakeWordGate:
    """Local wake word detection that mutes/unmutes the LiveKit mic track.

    Taps into the already-open LiveKit mic track via AudioStream — no
    second pyaudio stream needed (avoids ALSA "device unavailable").

    When enabled, the mic track starts muted. Audio frames are read from
    the LiveKit track and fed to an openWakeWord ONNX model. On detection,
    the mic is unmuted so the cloud agent hears speech. After a silence
    timeout (no agent speech for N seconds), the mic is re-muted.

    Room and avatar stay connected the entire time — only the mic toggles.
    """

    def __init__(self, model_path, threshold=0.5, silence_timeout=8.0):
        self._model_path = model_path
        self._threshold = threshold
        self._silence_timeout = silence_timeout
        self._mic_track = None
        self._room = None
        self._stop_event = threading.Event()
        self._last_agent_audio = 0.0
        self._mic_is_muted = False
        self._oww = None
        self._model_name = ""

    def start(self, mic_track, room):
        """Start wake word detection. Mutes the mic track immediately."""
        self._mic_track = mic_track
        self._room = room
        self._stop_event.clear()

        # Load model synchronously before starting async loop
        if not self._load_model():
            return

        # Mute mic on start — agent hears nothing until wake word
        self._set_mic_muted(True)
        queue_message("WAKEWORD: Mic muted — listening for wake word locally")

        # Start async detection loop (runs in the LiveKit event loop)
        asyncio.ensure_future(self._detection_loop())

    def stop(self):
        """Stop wake word detection."""
        self._stop_event.set()

    def notify_agent_audio(self):
        """Called when agent audio is received — resets silence timer."""
        self._last_agent_audio = time.monotonic()

    def _set_mic_muted(self, muted):
        """Mute or unmute the published mic track."""
        try:
            if muted:
                self._mic_track.mute()
            else:
                self._mic_track.unmute()
            self._mic_is_muted = muted
        except Exception as e:
            queue_message(f"WAKEWORD: Mute toggle error — {e}")

    def _load_model(self):
        """Load the openWakeWord model. Returns True on success."""
        import inspect

        try:
            from openwakeword.model import Model as OWWModel
        except ImportError:
            queue_message(
                "WAKEWORD: openwakeword not installed — "
                "wake word gate disabled. Install with: pip install openwakeword"
            )
            return False

        # Resolve model path — file path or pretrained name
        model_path = self._model_path
        if os.path.isfile(model_path):
            model_paths = [model_path]
        else:
            import openwakeword
            pretrained = openwakeword.get_pretrained_model_paths()

            def _name_matches(filepath, query):
                basename = os.path.basename(filepath).replace(".onnx", "").replace(".tflite", "")
                return basename == query or basename.startswith(query + "_v")

            match = next(
                (p for p in pretrained if _name_matches(p, model_path)),
                None,
            )
            if match:
                model_paths = [match]
            else:
                available = [os.path.basename(p).replace(".onnx", "") for p in pretrained]
                queue_message(
                    f"WAKEWORD: Model '{model_path}' not found. "
                    f"Available pretrained: {available}"
                )
                return False

        self._model_name = os.path.splitext(os.path.basename(model_paths[0]))[0]
        try:
            params = inspect.signature(OWWModel.__init__).parameters
            if "wakeword_models" in params:
                self._oww = OWWModel(wakeword_models=model_paths, inference_framework="onnx")
            else:
                self._oww = OWWModel(wakeword_model_paths=model_paths)
        except Exception as e:
            queue_message(f"WAKEWORD: Failed to load model — {e}")
            return False

        queue_message(f"WAKEWORD: Model loaded — '{self._model_name}', threshold={self._threshold}")
        return True

    async def _detection_loop(self):
        """Tap into the LiveKit mic track's audio for wake word detection.

        Uses AudioStream to read frames from the already-open mic —
        no second pyaudio stream needed.
        """
        _ensure_sdk()
        audio_stream = _rtc.AudioStream(self._mic_track, sample_rate=16000, num_channels=1)

        queue_message(f"WAKEWORD: Listening for '{self._model_name}'...")

        async for frame_event in audio_stream:
            if self._stop_event.is_set():
                break

            # Convert audio frame to int16 numpy for openWakeWord
            frame = frame_event.frame
            audio_int16 = np.frombuffer(frame.data, dtype=np.int16)

            # When mic is live — just check silence timeout
            if not self._mic_is_muted:
                elapsed = time.monotonic() - self._last_agent_audio
                if elapsed > self._silence_timeout:
                    logging.debug("WAKEWORD: Silence timeout — re-muting mic")
                    self._set_mic_muted(True)
                    set_tars_state(TarsState.STANDBY)
                    self._oww.reset()
                continue

            # Feed audio to openWakeWord
            self._oww.predict(audio_int16)

            # Check scores
            for name, scores in self._oww.prediction_buffer.items():
                if len(scores) > 0 and scores[-1] > self._threshold:
                    queue_message(
                        f"WAKEWORD: Detected '{name}' "
                        f"(score={scores[-1]:.2f}) — unmuting mic"
                    )
                    self._oww.reset()
                    set_tars_state(TarsState.LISTENING)
                    self._last_agent_audio = time.monotonic()
                    self._set_mic_muted(False)
                    break

        queue_message("WAKEWORD: Detection stopped")


# ── Lazy SDK import ──────────────────────────────────────────────────

_rtc = None


def _ensure_sdk():
    """Import the livekit RTC SDK on first use."""
    global _rtc
    if _rtc is not None:
        return
    try:
        from livekit import rtc
        _rtc = rtc
    except ImportError:
        raise ImportError(
            "livekit package is required for LiveKit mode. "
            "Install it with: pip install livekit"
        )


# ── Token generation ─────────────────────────────────────────────────

def _generate_token(room_name, identity):
    """Generate a LiveKit access token using the API key/secret."""
    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        raise ImportError(
            "livekit-api package is required for token generation. "
            "Install it with: pip install livekit-api"
        )

    lk_cfg = CONFIG["LIVEKIT"]
    api_key = lk_cfg["livekit_api_key"]
    api_secret = lk_cfg["livekit_api_secret"]

    if not api_key or not api_secret:
        raise ValueError(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env"
        )

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))
    )

    return token.to_jwt()


# ── LiveKit Client ───────────────────────────────────────────────────

class TarsLiveKitClient:
    """Pi joins a LiveKit room as a lightweight RTC participant.

    Publishes mic audio, receives agent audio/video, handles RPCs
    for physical actions (movement, servos, emotion, etc.).
    """

    def __init__(self, ui_manager=None):
        _ensure_sdk()

        # Room created in connect() — needs an active event loop
        self._room = None
        self._ui_manager = ui_manager
        self._connected = False
        self._shutdown = threading.Event()

        # Audio handled by Python SDK, video by browser or pygame
        self._media_devices = None
        self._mic_input = None
        self._mic_track = None
        self._audio_player = None

        # Wake word gate
        self._wake_word_gate = None

        # Config
        lk_cfg = CONFIG["LIVEKIT"]
        self._livekit_url = lk_cfg["livekit_url"]
        self._room_name = lk_cfg["room_name"]
        self._identity = lk_cfg["participant_identity"]
        self._play_local_audio = lk_cfg["play_local_audio"]
        self._display_mode = lk_cfg.get("display_mode", "browser")

    async def connect(self):
        """Create room with agent dispatch, then connect as participant."""
        if self._connected:
            return

        if not self._livekit_url:
            raise ValueError(
                "LIVEKIT_URL must be set in .env"
            )

        import uuid

        lk_cfg = CONFIG["LIVEKIT"]
        api_key = lk_cfg["livekit_api_key"]
        api_secret = lk_cfg["livekit_api_secret"]

        # Fresh room each session — avoids stale dispatch issues
        room_prefix = self._room_name
        self._room_name = f"{room_prefix}-{uuid.uuid4().hex[:8]}"

        self._room = _rtc.Room()
        token = _generate_token(self._room_name, self._identity)

        self._register_room_events()

        queue_message(f"LIVEKIT: Connecting to {self._livekit_url} "
                      f"room={self._room_name} as {self._identity}")

        # Don't auto-subscribe — we manually handle what we need
        # Browser mode: no tracks needed in Python (browser handles all media)
        # Pygame mode: need video tracks for rendering
        await self._room.connect(
            self._livekit_url, token,
            options=_rtc.RoomOptions(auto_subscribe=False),
        )
        self._connected = True
        queue_message("LIVEKIT: Connected to room")

        # Now explicitly dispatch the agent to this room
        try:
            from livekit.api import LiveKitAPI, CreateAgentDispatchRequest

            api = LiveKitAPI(
                url=self._livekit_url,
                api_key=api_key,
                api_secret=api_secret,
            )
            await api.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name="tars-agent",
                    room=self._room_name,
                )
            )
            await api.aclose()
            queue_message("LIVEKIT: Agent dispatched → tars-agent")
        except Exception as e:
            queue_message(f"LIVEKIT: Agent dispatch warning — {e}")

        set_tars_state(TarsState.STANDBY)

        # Publish mic (Python SDK — always, browser can't reliably access mic)
        await self._start_mic()

        # Audio output: Python SDK (play_local_audio=true) or browser (false)
        if self._play_local_audio:
            await self._start_audio_output()

        # Register RPC handlers
        self._register_rpc_handlers()

        # Launch video display
        if self._display_mode == "pygame":
            _start_pygame_display()
            queue_message("LIVEKIT: Client initialized — audio Python, video Pygame")
        else:
            _start_display_server(self._livekit_url, self._room_name)
            queue_message("LIVEKIT: Client initialized — audio Python, video browser")

        # Wake word gate (optional — mutes mic until wake word detected)
        if lk_cfg.get("wake_word_enabled", False):
            model_val = lk_cfg.get("wake_word_model", "hey_mycroft")
            # Resolve model path: pretrained name, relative path, or absolute path
            if os.path.isabs(model_val):
                model_path = model_val
            elif model_val.endswith(".onnx"):
                # Relative path from src/
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, model_val)
            else:
                # Pretrained model name — openWakeWord resolves it internally
                model_path = model_val
            self._wake_word_gate = WakeWordGate(
                model_path=model_path,
                threshold=lk_cfg.get("wake_word_threshold", 0.5),
                silence_timeout=lk_cfg.get("wake_word_silence_timeout", 8.0),
            )
            self._wake_word_gate.start(self._mic_track, self._room)
            queue_message("LIVEKIT: Wake word gate enabled")

    async def disconnect(self):
        """Disconnect from the room and clean up."""
        self._shutdown.set()

        if self._wake_word_gate is not None:
            self._wake_word_gate.stop()
            self._wake_word_gate = None

        _stop_display()

        if self._audio_player is not None:
            try:
                await self._audio_player.aclose()
            except Exception:
                pass
            self._audio_player = None

        if self._mic_input is not None:
            try:
                await self._mic_input.aclose()
            except Exception:
                pass
            self._mic_input = None

        if self._connected:
            await self._room.disconnect()
            self._connected = False
            queue_message("LIVEKIT: Disconnected from room")

    # ── Mic publishing ───────────────────────────────────────────

    async def _start_mic(self):
        """Open local mic via MediaDevices and publish as audio track."""
        self._media_devices = _rtc.MediaDevices()

        self._mic_input = self._media_devices.open_input(
            enable_aec=True,
            noise_suppression=True,
            auto_gain_control=True,
        )

        self._mic_track = _rtc.LocalAudioTrack.create_audio_track(
            "tars-mic", self._mic_input.source
        )

        options = _rtc.TrackPublishOptions(
            source=_rtc.TrackSource.SOURCE_MICROPHONE,
        )
        await self._room.local_participant.publish_track(
            self._mic_track, options
        )
        queue_message("LIVEKIT: Mic track published")

    # ── Audio output (agent → speaker) ───────────────────────────

    async def _start_audio_output(self):
        """Set up speaker output on the USB audio device."""
        import sounddevice as sd

        # Find real USB output — ALSA 'default' (128ch) doesn't work
        output_idx = None
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) < 1:
                continue
            if "usb" in dev.get("name", "").lower():
                output_idx = i
                queue_message(f"LIVEKIT: Audio output → {dev['name']} (device {i})")
                break

        apm = self._mic_input.apm if self._mic_input else None
        delay = self._mic_input.delay_estimator if self._mic_input else None
        self._audio_player = _rtc.media_devices.OutputPlayer(
            apm_for_reverse=apm,
            delay_estimator=delay,
            output_device=output_idx,
        )
        queue_message("LIVEKIT: Audio output ready")

    # ── Video receive (pygame mode) ──────────────────────────────

    async def _handle_video_track(self, track):
        """Receive video frames and push to pygame display at ~25fps."""
        _rtc_local = _rtc
        stream = _rtc_local.VideoStream(track)

        queue_message("LIVEKIT: Receiving video → pygame display")

        min_interval = 1.0 / 28  # slightly above 25fps target
        last_time = 0.0

        display = _pygame_display

        async for frame_event in stream:
            if self._shutdown.is_set():
                break

            now = time.monotonic()
            if now - last_time < min_interval:
                continue
            last_time = now

            frame = frame_event.frame
            rgba = frame.convert(_rtc_local.VideoBufferType.RGBA)
            arr = np.frombuffer(rgba.data, dtype=np.uint8)
            arr = arr.reshape((rgba.height, rgba.width, 4))

            if display is not None:
                display.set_frame(arr)

    # ── Room event handlers ──────────────────────────────────────

    def _register_room_events(self):
        """Register handlers for room events."""

        # With auto_subscribe=False, we manually subscribe to tracks we need
        @self._room.on("track_published")
        def on_track_published(
            publication: _rtc.RemoteTrackPublication,
            participant: _rtc.RemoteParticipant,
        ):
            # Subscribe to audio if Python handles it
            if publication.kind == _rtc.TrackKind.KIND_AUDIO and self._play_local_audio:
                publication.set_subscribed(True)
                queue_message(f"LIVEKIT: Subscribing to audio from {participant.identity}")

            # Subscribe to video only in pygame mode (browser subscribes itself)
            if publication.kind == _rtc.TrackKind.KIND_VIDEO and self._display_mode == "pygame":
                publication.set_subscribed(True)
                queue_message(f"LIVEKIT: Subscribing to video from {participant.identity}")

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: _rtc.Track,
            publication: _rtc.RemoteTrackPublication,
            participant: _rtc.RemoteParticipant,
        ):
            if track.kind == _rtc.TrackKind.KIND_AUDIO:
                if self._audio_player is not None:
                    async def _add_and_start(t, player):
                        await player.add_track(t)
                        try:
                            await player.start()
                        except RuntimeError:
                            pass
                    asyncio.ensure_future(
                        _add_and_start(track, self._audio_player)
                    )
                    queue_message(
                        f"LIVEKIT: Audio from {participant.identity} → speaker"
                    )
                # Notify wake word gate that agent is speaking
                if self._wake_word_gate is not None:
                    self._wake_word_gate.notify_agent_audio()
                set_tars_state(TarsState.TALKING)

            if track.kind == _rtc.TrackKind.KIND_VIDEO:
                asyncio.ensure_future(self._handle_video_track(track))
                queue_message(f"LIVEKIT: Video from {participant.identity} → pygame")

        @self._room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: _rtc.Track,
            publication: _rtc.RemoteTrackPublication,
            participant: _rtc.RemoteParticipant,
        ):
            queue_message(
                f"LIVEKIT: Track unsubscribed: {track.kind} "
                f"from {participant.identity}"
            )
            if track.kind == _rtc.TrackKind.KIND_AUDIO:
                set_tars_state(TarsState.STANDBY)

        @self._room.on("participant_connected")
        def on_participant_connected(participant: _rtc.RemoteParticipant):
            queue_message(
                f"LIVEKIT: Participant connected: {participant.identity}"
            )

        @self._room.on("participant_disconnected")
        def on_participant_disconnected(participant: _rtc.RemoteParticipant):
            queue_message(
                f"LIVEKIT: Participant disconnected: {participant.identity}"
            )
            set_tars_state(TarsState.STANDBY)

        @self._room.on("disconnected")
        def on_disconnected():
            queue_message("LIVEKIT: Disconnected from room")
            self._connected = False
            set_tars_state(TarsState.STANDBY)

    # ── RPC Handlers (agent → Pi physical actions) ───────────────

    def _register_rpc_handlers(self):
        """Expose physical actions as RPCs the cloud agent can call."""

        @self._room.local_participant.register_rpc_method("move")
        async def handle_move(data):
            """Execute a named movement sequence.
            Payload: {"name": "walk_forward", "speed": "slow"}
            """
            try:
                params = json.loads(data.payload)
                name = params.get("name", "")
                speed = params.get("speed", "slow")

                from modules.module_movements import (
                    walk_forward, walk_backward, step_forward, step_backward,
                    turn_left, turn_right, turn_left_slow, turn_right_slow,
                    neutral_legs, wave_right, wave_left,
                    right_hi, left_hi, bow, laugh, excited,
                    happy_dance, tilt_right, tilt_left,
                )

                movements = {
                    "walk_forward": walk_forward,
                    "walk_backward": walk_backward,
                    "step_forward": step_forward,
                    "step_backward": step_backward,
                    "turn_left": turn_left if speed == "fast" else turn_left_slow,
                    "turn_right": turn_right if speed == "fast" else turn_right_slow,
                    "neutral": neutral_legs,
                    "wave_right": wave_right,
                    "wave_left": wave_left,
                    "right_hi": right_hi,
                    "left_hi": left_hi,
                    "bow": bow,
                    "laugh": laugh,
                    "excited": excited,
                    "happy_dance": happy_dance,
                    "tilt_right": tilt_right,
                    "tilt_left": tilt_left,
                }

                func = movements.get(name)
                if func:
                    queue_message(f"LIVEKIT RPC: Moving — {name} ({speed})")
                    # Fire and forget — don't block the agent
                    import threading as _thr
                    _thr.Thread(target=func, daemon=True).start()
                    return json.dumps({"status": "ok", "movement": name})
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Unknown movement: {name}",
                        "available": list(movements.keys()),
                    })
            except Exception as e:
                queue_message(f"LIVEKIT RPC: Move error — {e}")
                return json.dumps({"status": "error", "message": str(e)})

        @self._room.local_participant.register_rpc_method("gesture")
        async def handle_gesture(data):
            """Execute a gesture animation.
            Payload: {"name": "nod"}
            """
            try:
                params = json.loads(data.payload)
                name = params.get("name", "")

                from modules.module_gestures import execute_gesture
                queue_message(f"LIVEKIT RPC: Gesture — {name}")
                import threading as _thr
                _thr.Thread(target=execute_gesture, args=(name,), daemon=True).start()
                return json.dumps({"status": "ok", "gesture": name})
            except Exception as e:
                queue_message(f"LIVEKIT RPC: Gesture error — {e}")
                return json.dumps({"status": "error", "message": str(e)})

        @self._room.local_participant.register_rpc_method("set_servo")
        async def handle_set_servo(data):
            """Set a specific servo to a PWM value.
            Payload: {"channel": 0, "value": 300}
            """
            try:
                params = json.loads(data.payload)
                channel = int(params["channel"])
                value = int(params["value"])

                from modules.module_servoctl import set_servo_pwm
                queue_message(f"LIVEKIT RPC: Servo ch{channel} → {value}")
                set_servo_pwm(channel, value)
                return json.dumps({"status": "ok"})
            except Exception as e:
                queue_message(f"LIVEKIT RPC: Servo error — {e}")
                return json.dumps({"status": "error", "message": str(e)})

        @self._room.local_participant.register_rpc_method("set_emotion")
        async def handle_set_emotion(data):
            """Set TARS face emotion on the display.
            Payload: {"emotion": "happy"}
            """
            try:
                params = json.loads(data.payload)
                emotion = params.get("emotion", "neutral")

                # Update ChatUI emotion display
                try:
                    from modules.module_chatui import update_emotion
                    update_emotion(emotion)
                except Exception:
                    pass

                # Update local UI
                if self._ui_manager:
                    self._ui_manager.update_data(
                        "System", f"Emotion: {emotion}", "SYSTEM"
                    )

                queue_message(f"LIVEKIT RPC: Emotion → {emotion}")
                return json.dumps({"status": "ok", "emotion": emotion})
            except Exception as e:
                queue_message(f"LIVEKIT RPC: Emotion error — {e}")
                return json.dumps({"status": "error", "message": str(e)})

        @self._room.local_participant.register_rpc_method("get_battery")
        async def handle_get_battery(data):
            """Get battery status.
            Payload: {} (empty)
            """
            try:
                from modules.module_battery import BatteryModule
                battery = BatteryModule()
                info = {
                    "percentage": battery.get_percentage(),
                    "voltage": battery.get_voltage(),
                    "is_charging": battery.is_charging(),
                }
                return json.dumps({"status": "ok", **info})
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                })

        @self._room.local_participant.register_rpc_method("disable_servos")
        async def handle_disable_servos(data):
            """Disable all servos (relax mode)."""
            try:
                from modules.module_servoctl import disable_all_servos
                disable_all_servos()
                queue_message("LIVEKIT RPC: All servos disabled")
                return json.dumps({"status": "ok"})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        @self._room.local_participant.register_rpc_method("set_state")
        async def handle_set_state(data):
            """Set TARS application state.
            Payload: {"state": "listening"}
            """
            try:
                params = json.loads(data.payload)
                state_name = params.get("state", "standby").upper()
                state = TarsState[state_name]
                set_tars_state(state)
                queue_message(f"LIVEKIT RPC: State → {state_name}")
                return json.dumps({"status": "ok", "state": state_name})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        queue_message(
            "LIVEKIT: RPC handlers registered — move, gesture, set_servo, "
            "set_emotion, get_battery, disable_servos, set_state"
        )

    @property
    def is_connected(self):
        return self._connected


# ── Module-level management ──────────────────────────────────────────

_client_instance = None
_client_lock = threading.Lock()


def get_client():
    """Get the global TarsLiveKitClient (create if needed)."""
    global _client_instance
    with _client_lock:
        if _client_instance is None:
            _client_instance = TarsLiveKitClient()
        return _client_instance


def start_livekit_client(ui_manager=None, shutdown_event=None):
    """Start the LiveKit client in a new asyncio event loop (blocking).

    Called from a daemon thread in app.py. Runs until shutdown_event is set.
    """
    global _client_instance

    client = TarsLiveKitClient(ui_manager=ui_manager)
    with _client_lock:
        _client_instance = client

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(client.connect())

        # Run until shutdown
        async def _wait_shutdown():
            while not (shutdown_event and shutdown_event.is_set()):
                await asyncio.sleep(0.5)
            await client.disconnect()

        loop.run_until_complete(_wait_shutdown())
    except Exception as e:
        queue_message(f"LIVEKIT: Client error — {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            loop.run_until_complete(client.disconnect())
        except Exception:
            pass
        loop.close()
        queue_message("LIVEKIT: Client stopped")


def stop_livekit_client():
    """Stop the global client (if running).

    The LiveKit client runs in its own asyncio loop on a daemon thread.
    Disconnect is handled by the shutdown_event in that loop's _wait_shutdown.
    This function just ensures the wake word gate is stopped and cleans up
    the reference — the daemon thread exits when the process exits.
    """
    global _client_instance
    with _client_lock:
        if _client_instance is not None:
            # Stop wake word gate (blocks on pyaudio stream)
            if _client_instance._wake_word_gate is not None:
                _client_instance._wake_word_gate.stop()
            _client_instance = None
