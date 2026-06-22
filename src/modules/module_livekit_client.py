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
import threading
import time
import numpy as np

from modules.module_config import load_config
from modules.module_messageQue import queue_message
from modules.module_state import set_tars_state, TarsState

CONFIG = load_config()

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

        # Track references
        self._media_devices = None
        self._mic_input = None
        self._mic_track = None
        self._audio_player = None
        self._video_stream = None

        # Config
        lk_cfg = CONFIG["LIVEKIT"]
        self._livekit_url = lk_cfg["livekit_url"]
        self._room_name = lk_cfg["room_name"]
        self._identity = lk_cfg["participant_identity"]
        self._play_local_audio = lk_cfg["play_local_audio"]

    async def connect(self):
        """Connect to LiveKit Cloud room and set up tracks + RPCs."""
        if self._connected:
            return

        if not self._livekit_url:
            raise ValueError(
                "LIVEKIT_URL must be set in .env"
            )

        # Create room inside async context (requires active event loop)
        self._room = _rtc.Room()

        token = _generate_token(self._room_name, self._identity)

        # Register event handlers before connecting
        self._register_room_events()

        queue_message(f"LIVEKIT: Connecting to {self._livekit_url} "
                      f"room={self._room_name} as {self._identity}")

        await self._room.connect(self._livekit_url, token)
        self._connected = True
        queue_message("LIVEKIT: Connected to room")

        set_tars_state(TarsState.STANDBY)

        # Publish mic
        await self._start_mic()

        # Set up audio output for receiving agent audio
        if self._play_local_audio:
            await self._start_audio_output()

        # Register RPC handlers
        self._register_rpc_handlers()

        queue_message("LIVEKIT: Client fully initialized — mic publishing, "
                      "audio output ready, RPCs registered")

    async def disconnect(self):
        """Disconnect from the room and clean up."""
        self._shutdown.set()

        if self._video_stream is not None:
            try:
                self._video_stream.close()
            except Exception:
                pass
            self._video_stream = None

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
        """Set up speaker output with AEC linked to mic input."""
        # Pass the mic's APM for echo cancellation
        apm = self._mic_input.apm if self._mic_input else None
        delay = self._mic_input.delay_estimator if self._mic_input else None
        self._audio_player = _rtc.media_devices.OutputPlayer(
            apm_for_reverse=apm,
            delay_estimator=delay,
        )
        queue_message("LIVEKIT: Audio output device opened")

    # ── Video receive (agent → Pygame display) ───────────────────

    async def _handle_video_track(self, track):
        """Receive video frames from agent and render on Pygame display."""
        stream = _rtc.VideoStream(track)
        self._video_stream = stream

        queue_message("LIVEKIT: Receiving agent video stream")

        async for frame_event in stream:
            if self._shutdown.is_set():
                break

            frame = frame_event.frame
            # Convert to RGBA for pygame
            rgba_frame = frame.convert(_rtc.VideoBufferType.RGBA)
            frame_data = np.frombuffer(rgba_frame.data, dtype=np.uint8)
            frame_data = frame_data.reshape(
                (rgba_frame.height, rgba_frame.width, 4)
            )

            # Push to UI manager if available
            if self._ui_manager and hasattr(self._ui_manager, 'set_livekit_frame'):
                self._ui_manager.set_livekit_frame(frame_data)

    # ── Room event handlers ──────────────────────────────────────

    def _register_room_events(self):
        """Register handlers for room events."""

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: _rtc.Track,
            publication: _rtc.RemoteTrackPublication,
            participant: _rtc.RemoteParticipant,
        ):
            queue_message(
                f"LIVEKIT: Track subscribed: {track.kind} "
                f"from {participant.identity}"
            )

            if track.kind == _rtc.TrackKind.KIND_AUDIO:
                # Route agent audio to local speaker
                if self._audio_player is not None:
                    self._audio_player.add_track(track)
                    asyncio.ensure_future(self._audio_player.start())
                    queue_message("LIVEKIT: Agent audio → speaker")
                set_tars_state(TarsState.TALKING)

            elif track.kind == _rtc.TrackKind.KIND_VIDEO:
                # Route agent video to Pygame display
                asyncio.ensure_future(self._handle_video_track(track))

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
                    neutral_legs,
                )

                movements = {
                    "walk_forward": walk_forward,
                    "walk_backward": walk_backward,
                    "step_forward": step_forward,
                    "step_backward": step_backward,
                    "turn_left": turn_left if speed == "fast" else turn_left_slow,
                    "turn_right": turn_right if speed == "fast" else turn_right_slow,
                    "neutral": neutral_legs,
                }

                func = movements.get(name)
                if func:
                    queue_message(f"LIVEKIT RPC: Moving — {name} ({speed})")
                    func()
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
                execute_gesture(name)
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
    """Stop the global client (if running)."""
    global _client_instance
    with _client_lock:
        if _client_instance is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_client_instance.disconnect())
                loop.close()
            except Exception as e:
                queue_message(f"LIVEKIT: Error stopping client — {e}")
            _client_instance = None
