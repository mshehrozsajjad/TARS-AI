"""
Module: Deepgram STT
Real-time speech-to-text using the official Deepgram Python SDK (v2 API).

Streams mic audio to Deepgram in real-time while using local VAD for
end-of-speech detection.

Connection strategy: after each utterance, a new WebSocket is pre-opened
in the background so the next call finds it ready (~0ms vs ~500ms).

End-of-speech uses TWO parallel paths (whichever fires first):
  1. Deepgram end-of-turn confidence (fast, ~0.3-0.5s after last word)
  2. Local VAD silence timeout (slow fallback, speechdelay × 250ms)
"""

import os
import time
import threading

from modules.module_messageQue import queue_message

# ---------------------------------------------------------------------------
#  Connection pool — pre-opened connection ready for the next utterance
# ---------------------------------------------------------------------------
_client = None
_pending_ctx = None        # context manager for the pre-opened connection
_pending_conn = None       # the pre-opened connection object
_pending_listener = None   # listener thread for the pre-opened connection
_pending_lock = threading.Lock()
_pending_ready = threading.Event()

# Per-utterance mutable state — written by listener thread, read by main
_transcript = None
_done = threading.Event()
_eot = threading.Event()

# Deepgram end-of-turn confidence threshold (0.0–1.0).
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
    """Message handler — updates per-utterance state."""
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


def _create_connection():
    """Create a new WebSocket connection + listener thread. Returns (ctx, conn, listener) or None."""
    global _client

    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        print("[DEEPGRAM] ERROR: DEEPGRAM_API_KEY not set", flush=True)
        return None

    try:
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
    except ImportError:
        print("[DEEPGRAM] ERROR: pip install deepgram-sdk", flush=True)
        return None

    if _client is None:
        _client = DeepgramClient(api_key=api_key)

    try:
        ctx = _client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=16000,
        )
        conn = ctx.__enter__()
        conn.on(EventType.MESSAGE, _on_message)
        conn.on(EventType.ERROR, _on_error)

        listener = threading.Thread(target=conn.start_listening, daemon=True)
        listener.start()
        return ctx, conn, listener

    except Exception as e:
        print(f"[DEEPGRAM] Connection failed: {e}", flush=True)
        return None


def _preconnect():
    """Open a connection in the background so it's ready for the next utterance."""
    global _pending_ctx, _pending_conn, _pending_listener
    with _pending_lock:
        if _pending_conn is not None:
            return  # Already have one ready
    result = _create_connection()
    if result:
        ctx, conn, listener = result
        with _pending_lock:
            _pending_ctx, _pending_conn, _pending_listener = ctx, conn, listener
            _pending_ready.set()


def _take_connection():
    """Take the pre-opened connection, or create a new one if none ready."""
    global _pending_ctx, _pending_conn, _pending_listener
    with _pending_lock:
        if _pending_conn is not None:
            ctx, conn, listener = _pending_ctx, _pending_conn, _pending_listener
            _pending_ctx = _pending_conn = _pending_listener = None
            _pending_ready.clear()
            return ctx, conn, listener

    # No pre-opened connection — create one now
    result = _create_connection()
    if result:
        return result
    return None


def _close_connection(ctx):
    """Cleanly close a connection's context manager."""
    try:
        ctx.__exit__(None, None, None)
    except Exception:
        pass


def transcribe_streaming(stt_manager):
    """Stream mic audio to Deepgram while using local VAD + Deepgram EOT.

    First call pays ~500ms for WebSocket setup. Subsequent calls find a
    pre-opened connection ready (~0ms).

    Args:
        stt_manager: The STTManager instance (for VAD, mic, thresholds).

    Returns:
        str: Final transcript text, or None.
    """
    global _transcript

    t_start = time.monotonic()

    taken = _take_connection()
    if taken is None:
        return None
    ctx, connection, listener = taken

    t_connected = time.monotonic()
    connect_ms = (t_connected - t_start) * 1000
    if connect_ms > 50:
        print(f"[DEEPGRAM] Connected in {connect_ms:.0f}ms", flush=True)
    else:
        print(f"[DEEPGRAM] Reused pre-opened connection ({connect_ms:.0f}ms)", flush=True)

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
    aborted = False

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
                    aborted = True
                    break

                # Send audio to Deepgram in real-time
                try:
                    connection.send_media(stt_manager.amplify_audio(data).tobytes())
                except Exception as e:
                    print(f"[DEEPGRAM] send_media failed: {e}", flush=True)
                    break

                # Run local VAD
                is_silence, detected_speech, silent_frames = vad_func(data, detected_speech, silent_frames)

                # ----- Fast path: Deepgram end-of-turn confidence -----
                if _eot.is_set() and detected_speech and speech_frames >= min_speech_frames:
                    print(f"[DEEPGRAM] EOT (Deepgram) after {speech_frames} speech frames "
                          f"({(time.monotonic() - t_connected)*1000:.0f}ms)", flush=True)
                    break

                # ----- Slow fallback: local VAD silence timeout -----
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
                        print(f"[DEEPGRAM] EOT (VAD) after {speech_frames} speech frames", flush=True)
                        break

                if detected_speech and not is_silence:
                    if speech_frames == 0:
                        print("[DEEPGRAM] Speech detected, streaming...", flush=True)
                    speech_frames += 1

    except Exception as e:
        print(f"[DEEPGRAM] Recording error: {e}", flush=True)

    # Close this connection and pre-open the next one in the background
    t_vad_done = time.monotonic()
    try:
        connection.send_close_stream()
    except Exception:
        pass
    threading.Thread(target=_close_connection, args=(ctx,), daemon=True).start()
    threading.Thread(target=_preconnect, daemon=True).start()

    if aborted or speech_frames < min_speech_frames:
        return None

    # Wait for final transcript if we don't have one yet
    if not _transcript:
        _done.wait(timeout=5.0)

    t_result = time.monotonic()
    result = _transcript

    if result:
        print(f"[DEEPGRAM] Final ({(t_result - t_vad_done)*1000:.0f}ms): {result}", flush=True)
    else:
        print(f"[DEEPGRAM] No transcript ({(t_result - t_vad_done)*1000:.0f}ms waited)", flush=True)

    return result
