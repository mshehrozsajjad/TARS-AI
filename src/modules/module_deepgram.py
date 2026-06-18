"""
Module: Deepgram STT
Real-time speech-to-text using the official Deepgram Python SDK (v2 API).

Streams mic audio to Deepgram in real-time while using local VAD for
end-of-speech detection.  A persistent WebSocket connection is kept
alive across utterances so that only the first call pays the ~500ms
handshake cost; subsequent calls reuse the open connection and signal
utterance boundaries with send_finalize().

End-of-speech is detected via TWO parallel paths (whichever fires first):
  1. Deepgram end-of-turn confidence (fast, ~0.3-0.5s after last word)
  2. Local VAD silence timeout (slow fallback, speechdelay × 250ms)
"""

import os
import time
import threading

from modules.module_messageQue import queue_message

# ---------------------------------------------------------------------------
#  Persistent connection state (module-level, survives across utterances)
# ---------------------------------------------------------------------------
_client = None
_ctx_manager = None       # context manager returned by listen.v2.connect()
_connection = None         # the actual connection object
_listener = None           # daemon thread running start_listening()

# Per-utterance mutable state — written by listener thread, read by main
_transcript = None         # latest non-empty transcript from Deepgram
_done = threading.Event()  # set when a final/usable transcript is ready
_eot = threading.Event()   # set when Deepgram end-of-turn confidence is high

# Deepgram end-of-turn confidence threshold (0.0–1.0).
# Lower = faster but more false positives.  0.5 is Deepgram's recommended default.
EOT_THRESHOLD = 0.5


def _extract_transcript(message):
    """Extract transcript text from a Deepgram v2 message (plain dict)."""
    if not isinstance(message, dict):
        return getattr(message, "transcript", None)
    if "transcript" in message:
        return message["transcript"]
    if "channel" in message:
        try:
            return message["channel"]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            pass
    return None


def _on_message(message):
    """Global message handler for the persistent connection."""
    global _transcript
    if not isinstance(message, dict):
        return
    msg_type = message.get("type", "")
    if msg_type in ("Connected", "Metadata"):
        return

    text = _extract_transcript(message)
    if text and text.strip():
        _transcript = text.strip()

    # Use Deepgram's end-of-turn confidence for fast end-of-speech
    eot_conf = message.get("end_of_turn_confidence", 0)
    if eot_conf >= EOT_THRESHOLD and _transcript:
        _eot.set()
        _done.set()


def _on_error(error):
    print(f"[DEEPGRAM] Error: {error}", flush=True)
    _done.set()


def _on_close(_):
    """Connection was closed — mark stale so next call reconnects."""
    global _connection, _ctx_manager, _listener
    _connection = None
    _ctx_manager = None
    _listener = None
    _done.set()


def _open_connection():
    """Open a new persistent WebSocket and start the listener thread."""
    global _client, _ctx_manager, _connection, _listener

    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        print("[DEEPGRAM] ERROR: DEEPGRAM_API_KEY not set", flush=True)
        return False

    try:
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
    except ImportError:
        print("[DEEPGRAM] ERROR: pip install deepgram-sdk", flush=True)
        return False

    if _client is None:
        _client = DeepgramClient(api_key=api_key)

    try:
        _ctx_manager = _client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=16000,
        )
        _connection = _ctx_manager.__enter__()

        _connection.on(EventType.MESSAGE, _on_message)
        _connection.on(EventType.ERROR, _on_error)
        _connection.on(EventType.CLOSE, _on_close)

        _listener = threading.Thread(target=_connection.start_listening, daemon=True)
        _listener.start()
        return True

    except Exception as e:
        print(f"[DEEPGRAM] Connection failed: {e}", flush=True)
        _connection = None
        _ctx_manager = None
        _listener = None
        return False


def _ensure_connection():
    """Return True if a usable connection exists (opening one if needed)."""
    if _connection is not None:
        return True
    return _open_connection()


def close_connection():
    """Cleanly shut down the persistent connection (call on STT shutdown)."""
    global _connection, _ctx_manager, _listener
    if _ctx_manager is not None:
        try:
            _ctx_manager.__exit__(None, None, None)
        except Exception:
            pass
    _connection = None
    _ctx_manager = None
    _listener = None


def transcribe_streaming(stt_manager):
    """Stream mic audio to Deepgram while using local VAD + Deepgram EOT
    for end-of-speech detection.

    Uses the persistent WebSocket connection — first call pays ~500ms,
    subsequent calls start instantly.

    Args:
        stt_manager: The STTManager instance (for VAD, mic, thresholds).

    Returns:
        str: Final transcript text, or None.
    """
    global _transcript

    t_start = time.monotonic()

    if not _ensure_connection():
        return None

    t_connected = time.monotonic()
    connect_ms = (t_connected - t_start) * 1000
    if connect_ms > 50:
        print(f"[DEEPGRAM] Connected in {connect_ms:.0f}ms", flush=True)

    from modules.module_mic import ResamplingInputStream
    from modules.module_tts import is_tts_playing, needs_mic_flush, clear_mic_flush
    from modules.module_state import set_tars_state, TarsState

    # Reset per-utterance state
    _transcript = None
    _done.clear()
    _eot.clear()

    # Get VAD function from STT manager
    vad_dispatch = {
        "silero": stt_manager._is_silence_detected_silero,
        "sherpa-onnx": stt_manager._is_silence_detected_sherpa_onnx,
        "smart-turn": stt_manager._is_silence_detected_smart_turn
            if stt_manager.smart_turn_session is not None
            else stt_manager._is_silence_detected_rms,
    }
    vad_func = vad_dispatch.get(stt_manager.vadmethod, stt_manager._is_silence_detected_rms)

    # Reset VAD state
    if stt_manager.sherpa_vad is not None:
        stt_manager.sherpa_vad.reset()
    stt_manager.smart_turn_audio_buffer.clear()

    detected_speech = False
    silent_frames = 0
    speech_frames = 0
    max_silent_pre_speech = stt_manager.MAX_SILENT_FRAMES
    max_silent_post_speech = stt_manager.MAX_SILENT_FRAMES
    min_speech_frames = 5

    print(f"[DEEPGRAM] Listening (vad={stt_manager.vadmethod})", flush=True)

    try:
        with ResamplingInputStream(dtype="int16") as mic:
            # Flush stale mic audio from TTS
            try:
                if needs_mic_flush():
                    mic.flush()
                    clear_mic_flush()
            except Exception:
                pass

            for frame_idx in range(stt_manager.MAX_RECORDING_FRAMES):
                data, _ = mic.read(4000)

                # Abort if TTS started
                if is_tts_playing():
                    set_tars_state(TarsState.STANDBY)
                    print("[DEEPGRAM] Aborting — TTS started", flush=True)
                    try:
                        _connection.send_finalize()
                    except Exception:
                        pass
                    return None

                # Send audio to Deepgram in real-time
                try:
                    _connection.send_media(stt_manager.amplify_audio(data).tobytes())
                except Exception as e:
                    print(f"[DEEPGRAM] send_media failed: {e}", flush=True)
                    close_connection()
                    return None

                # Run local VAD
                is_silence, detected_speech, silent_frames = vad_func(data, detected_speech, silent_frames)

                # ----- Fast end-of-turn via Deepgram confidence -----
                if _eot.is_set() and detected_speech and speech_frames >= min_speech_frames:
                    t_eot = time.monotonic()
                    print(f"[DEEPGRAM] EOT detected by Deepgram after {speech_frames} frames "
                          f"({(t_eot - t_connected)*1000:.0f}ms)", flush=True)
                    break

                # ----- Slow fallback: local VAD silence timeout -----
                # Extend pre-speech timeout while a gesture is running
                _gesture_running = False
                try:
                    from modules.module_gestures import gesture_active
                    _gesture_running = gesture_active
                except Exception:
                    pass
                if not detected_speech and silent_frames >= max_silent_pre_speech and not _gesture_running:
                    break  # No speech detected, give up

                if is_silence and detected_speech and speech_frames >= min_speech_frames:
                    if silent_frames >= max_silent_post_speech:
                        print(f"[DEEPGRAM] End of speech (VAD) after {speech_frames} frames", flush=True)
                        break

                if detected_speech and not is_silence:
                    if speech_frames == 0:
                        print("[DEEPGRAM] Speech detected, streaming...", flush=True)
                    speech_frames += 1

    except Exception as e:
        print(f"[DEEPGRAM] Recording error: {e}", flush=True)
        close_connection()
        return None

    # Finalize current utterance (keeps the connection open for next time)
    t_vad_done = time.monotonic()
    try:
        _connection.send_finalize()
    except Exception as e:
        print(f"[DEEPGRAM] send_finalize failed: {e}", flush=True)
        close_connection()
        return None

    if speech_frames < min_speech_frames:
        return None

    # Wait for final transcript if we don't have one yet
    if not _transcript:
        _done.wait(timeout=5.0)

    t_result = time.monotonic()
    result = _transcript

    if result:
        print(f"[DEEPGRAM] Final ({(t_result - t_vad_done)*1000:.0f}ms after finalize): {result}", flush=True)
    else:
        print(f"[DEEPGRAM] No transcript ({(t_result - t_vad_done)*1000:.0f}ms waited)", flush=True)

    return result
