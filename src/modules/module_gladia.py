"""
Module: Gladia STT
Real-time speech-to-text using the official Gladia SDK.

Maintains a persistent WebSocket session across utterances within a
conversation to eliminate per-utterance connection overhead (~300-800ms).
Session is created on first use after wake word and kept alive through
follow-up utterances. Torn down when the conversation ends (sleep).

Local VAD is still used for:
- No-speech timeout (return to sleep if nobody speaks)
- End-of-speech detection (stop reading mic after silence)

After end-of-speech, Gladia's server-side processing delivers the final
transcript almost instantly since audio was streamed in real-time.
"""

import os
import queue
import threading
import time

from modules.module_messageQue import queue_message


# === Persistent Session State ===
_client = None
_session = None
_transcript_queue = None
_session_alive = False
_session_lock = threading.Lock()


def _create_session():
    """Create a new Gladia live session. Returns True on success."""
    global _client, _session, _transcript_queue, _session_alive

    try:
        from gladiaio_sdk import (
            GladiaClient,
            LiveV2InitRequest,
            LiveV2LanguageConfig,
            LiveV2MessagesConfig,
            LiveV2WebSocketMessage,
            LiveV2EndedMessage,
        )
    except ImportError:
        print("[GLADIA] ERROR: pip install gladiaio-sdk", flush=True)
        return False

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        print("[GLADIA] ERROR: GLADIA_API_KEY not set", flush=True)
        return False

    # Reuse client instance across sessions
    if _client is None:
        _client = GladiaClient(api_key=api_key)

    live = _client.live()
    _transcript_queue = queue.Queue()

    t0 = time.perf_counter()
    try:
        _session = live.start_session(
            LiveV2InitRequest(
                model="solaria-1",
                encoding="wav/pcm",
                sample_rate=16000,
                bit_depth=16,
                channels=1,
                language_config=LiveV2LanguageConfig(languages=["en"]),
                messages_config=LiveV2MessagesConfig(
                    receive_partial_transcripts=False,
                ),
            )
        )
    except Exception as e:
        print(f"[GLADIA] Failed to create session: {e}", flush=True)
        return False

    elapsed = time.perf_counter() - t0
    print(f"[GLADIA] Session created in {elapsed:.2f}s", flush=True)
    _session_alive = True

    @_session.on("message")
    def on_message(msg: LiveV2WebSocketMessage):
        if msg.type == "transcript" and msg.data.is_final:
            text = msg.data.utterance.text.strip()
            if text:
                _transcript_queue.put(text)

    @_session.on("error")
    def on_error(err: Exception):
        global _session_alive
        print(f"[GLADIA] Session error: {err}", flush=True)
        _session_alive = False
        try:
            _transcript_queue.put(None)
        except Exception:
            pass

    @_session.once("ended")
    def on_ended(msg: LiveV2EndedMessage):
        global _session_alive
        print("[GLADIA] Session ended by server", flush=True)
        _session_alive = False
        try:
            _transcript_queue.put(None)
        except Exception:
            pass

    return True


def _ensure_session():
    """Ensure a Gladia session is alive, creating one if needed.
    Must be called inside _session_lock."""
    global _session, _session_alive

    if _session is not None and _session_alive:
        # Drain stale transcripts from previous utterance
        while _transcript_queue is not None:
            try:
                _transcript_queue.get_nowait()
            except queue.Empty:
                break
        return True

    # Session is dead or doesn't exist
    _session = None
    _session_alive = False
    return _create_session()


def stop_session():
    """Tear down the persistent Gladia session.
    Called when conversation ends (going to sleep) or on shutdown."""
    global _session, _session_alive
    with _session_lock:
        if _session is not None:
            if _session_alive:
                try:
                    _session.stop_recording()
                except Exception as e:
                    print(f"[GLADIA] Error stopping session: {e}", flush=True)
            _session_alive = False
            _session = None
            print("[GLADIA] Session stopped", flush=True)


def transcribe_streaming(stt_manager):
    """Stream mic audio to persistent Gladia session with local VAD.

    Reuses the Gladia WebSocket across utterances within a conversation.
    Creates a new session on first call or if the previous one died.

    Args:
        stt_manager: The STTManager instance (for VAD, mic, config).

    Returns:
        str: Final transcript text, or None.
    """
    with _session_lock:
        if not _ensure_session():
            return None

    from modules.module_mic import ResamplingInputStream
    from modules.module_tts import is_tts_playing, needs_mic_flush, clear_mic_flush
    from modules.module_state import set_tars_state, TarsState

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

    print(f"[DEBUG] Gladia: streaming audio (persistent session), vad={stt_manager.vadmethod}", flush=True)

    with ResamplingInputStream(dtype="int16") as mic:
        # Flush stale mic audio from TTS
        try:
            if needs_mic_flush():
                mic.flush()
                clear_mic_flush()
        except Exception:
            pass

        for _ in range(stt_manager.MAX_RECORDING_FRAMES):
            data, _ = mic.read(4000)

            # Abort if TTS started (barge-in or error)
            if is_tts_playing():
                set_tars_state(TarsState.STANDBY)
                return None  # Session stays alive for next utterance

            # Send audio to persistent Gladia session
            if _session_alive and _session is not None:
                try:
                    _session.send_audio(stt_manager.amplify_audio(data).tobytes())
                except Exception as e:
                    print(f"[GLADIA] send_audio failed: {e}", flush=True)
                    stop_session()
                    break

            # Run local VAD
            is_silence, detected_speech, silent_frames = vad_func(data, detected_speech, silent_frames)

            if not detected_speech and silent_frames >= max_silent_pre_speech:
                break  # No speech detected, give up

            if is_silence and detected_speech and speech_frames >= min_speech_frames:
                if silent_frames >= max_silent_post_speech:
                    break  # End of speech

            if detected_speech and not is_silence:
                if speech_frames == 0:
                    print("[GLADIA] Speech detected, streaming...", flush=True)
                speech_frames += 1

    if speech_frames < min_speech_frames:
        return None

    # Wait for Gladia's final transcript.
    # Audio including trailing silence was already streamed in real-time,
    # so the transcript should arrive very quickly.
    try:
        transcript = _transcript_queue.get(timeout=5.0)
    except queue.Empty:
        print("[GLADIA] Timeout waiting for transcript after speech", flush=True)
        # Session might be stuck — tear down for recreation on next call
        stop_session()
        return None

    if transcript is None:
        # Session error or ended unexpectedly during wait
        return None

    print(f"[GLADIA] {transcript}", flush=True)
    return transcript
