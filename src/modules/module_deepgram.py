"""
Module: Deepgram STT
Real-time speech-to-text using the official Deepgram Python SDK (v2 API).

Streams mic audio to Deepgram in real-time while using local VAD for
end-of-speech detection. Audio is sent as the user speaks, so the
transcript is ready almost instantly after send_close_stream() is called.

DeepgramClient is reused across sessions to avoid re-initialization.
"""

import os
import threading

from modules.module_messageQue import queue_message


# Reuse client across sessions to avoid re-initialization
_client = None


def _extract_transcript(message):
    """Extract transcript text from a Deepgram v2 message (dict or object)."""
    # Messages arrive as plain dicts from the SDK
    if isinstance(message, dict):
        msg_type = message.get("type", "")

        # v2 TurnInfo: {"type": "TurnInfo", "transcript": "...", ...}
        if "transcript" in message:
            return message["transcript"]

        # v2 nested channel structure
        if "channel" in message:
            try:
                return message["channel"]["alternatives"][0]["transcript"]
            except (KeyError, IndexError, TypeError):
                pass

        # v2 turn_info wrapper
        if "turn_info" in message:
            ti = message["turn_info"]
            if isinstance(ti, dict) and "transcript" in ti:
                return ti["transcript"]

        # v2 data wrapper
        if "data" in message:
            d = message["data"]
            if isinstance(d, dict) and "transcript" in d:
                return d["transcript"]

        return None

    # Fallback: SDK object with attributes
    text = getattr(message, "transcript", None)
    if text is not None:
        return text

    try:
        channel = getattr(message, "channel", None)
        if channel:
            return channel.alternatives[0].transcript
    except Exception:
        pass

    return None


def transcribe_streaming(stt_manager):
    """Stream mic audio to Deepgram while using local VAD for end-of-speech.

    Creates a per-utterance v2 WebSocket session. Audio is streamed in
    real-time and finalized via send_close_stream() when VAD detects
    end-of-speech.

    Args:
        stt_manager: The STTManager instance (for VAD, mic, thresholds).

    Returns:
        str: Final transcript text, or None.
    """
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

    from modules.module_mic import ResamplingInputStream
    from modules.module_tts import is_tts_playing, needs_mic_flush, clear_mic_flush
    from modules.module_state import set_tars_state, TarsState

    # Reuse client across sessions
    if _client is None:
        _client = DeepgramClient(api_key=api_key)

    final_transcript = None
    done = threading.Event()

    with _client.listen.v2.connect(
        model="flux-general-en",
        encoding="linear16",
        sample_rate=16000,
    ) as connection:

        def on_message(message):
            nonlocal final_transcript
            msg_type = message.get("type", "Unknown") if isinstance(message, dict) else getattr(message, "type", "Unknown")
            print(f"[DEEPGRAM] Message: type={msg_type}", flush=True)

            # Skip connection/metadata messages
            if msg_type in ("Connected", "Metadata"):
                return

            # Debug: log full message for non-trivial types
            print(f"[DEEPGRAM] DEBUG payload: {message}", flush=True)

            text = _extract_transcript(message)
            if text and text.strip():
                final_transcript = text.strip()
                print(f"[DEEPGRAM] Transcript: {final_transcript}", flush=True)
                done.set()

        def on_error(error):
            print(f"[DEEPGRAM] Error: {error}", flush=True)
            done.set()

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)
        connection.on(EventType.CLOSE, lambda _: done.set())

        connection.start_listening()

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

        print(f"[DEEPGRAM] Starting stream, vad={stt_manager.vadmethod}", flush=True)

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
                    print("[DEEPGRAM] Aborting — TTS started playing", flush=True)
                    connection.send_close_stream()
                    done.wait(timeout=2)
                    return None

                # Send audio to Deepgram in real-time
                connection.send_media(stt_manager.amplify_audio(data).tobytes())

                # Run local VAD
                is_silence, detected_speech, silent_frames = vad_func(data, detected_speech, silent_frames)

                # Extend pre-speech timeout while a gesture is running
                _gesture_running = False
                try:
                    from modules.module_gestures import gesture_active
                    _gesture_running = gesture_active
                except Exception:
                    pass
                if not detected_speech and silent_frames >= max_silent_pre_speech and not _gesture_running:
                    print(f"[DEEPGRAM] No speech detected after {frame_idx+1} frames, giving up", flush=True)
                    break

                if is_silence and detected_speech and speech_frames >= min_speech_frames:
                    if silent_frames >= max_silent_post_speech:
                        print(f"[DEEPGRAM] End of speech after {speech_frames} frames", flush=True)
                        break

                if detected_speech and not is_silence:
                    if speech_frames == 0:
                        print("[DEEPGRAM] Speech detected, streaming...", flush=True)
                    speech_frames += 1

        # Signal end of audio — Deepgram finalizes the transcript
        connection.send_close_stream()

        if speech_frames < min_speech_frames:
            print(f"[DEEPGRAM] Too few speech frames ({speech_frames}), discarding", flush=True)
            done.wait(timeout=2)
            return None

        # Wait for final transcript — should arrive quickly since
        # Deepgram already received all audio in real-time
        done.wait(timeout=10.0)

    if final_transcript:
        print(f"[DEEPGRAM] Final: {final_transcript}", flush=True)
    else:
        print("[DEEPGRAM] No transcript received", flush=True)

    return final_transcript
