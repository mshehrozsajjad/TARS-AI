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
            msg_type = getattr(message, "type", "Unknown")
            msg_class = type(message).__name__
            print(f"[DEEPGRAM] Message: type={msg_type}, class={msg_class}", flush=True)

            # Log structure of every message for debugging
            try:
                attrs = [a for a in dir(message) if not a.startswith('_')]
                print(f"[DEEPGRAM] DEBUG attrs: {attrs}", flush=True)
                print(f"[DEEPGRAM] DEBUG repr: {message}", flush=True)
            except Exception:
                pass

            # Try to extract transcript from any message type
            text = None

            # Path 1: direct .transcript attribute
            text = getattr(message, "transcript", None)

            # Path 2: nested channel structure
            if text is None:
                try:
                    channels = getattr(message, "channels", None) or getattr(message, "channel", None)
                    if channels:
                        if isinstance(channels, list):
                            text = channels[0].alternatives[0].transcript
                        else:
                            text = channels.alternatives[0].transcript
                except Exception:
                    pass

            # Path 3: turn_info nested object
            if text is None:
                try:
                    turn_info = getattr(message, "turn_info", None)
                    if turn_info:
                        text = getattr(turn_info, "transcript", None)
                except Exception:
                    pass

            # Path 4: data nested object
            if text is None:
                try:
                    data = getattr(message, "data", None)
                    if data:
                        text = getattr(data, "transcript", None)
                except Exception:
                    pass

            if text and text.strip():
                final_transcript = text.strip()
                print(f"[DEEPGRAM] Extracted transcript: {final_transcript}", flush=True)
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

        print(f"[DEBUG] Deepgram: starting stream, vad={stt_manager.vadmethod}", flush=True)

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

                # Abort if TTS started
                if is_tts_playing():
                    set_tars_state(TarsState.STANDBY)
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
                    break  # No speech detected, give up

                if is_silence and detected_speech and speech_frames >= min_speech_frames:
                    if silent_frames >= max_silent_post_speech:
                        break  # End of speech

                if detected_speech and not is_silence:
                    if speech_frames == 0:
                        print("[DEEPGRAM] Speech detected, streaming...", flush=True)
                    speech_frames += 1

        # Signal end of audio — Deepgram finalizes the transcript
        connection.send_close_stream()

        if speech_frames < min_speech_frames:
            done.wait(timeout=2)
            return None

        # Wait for final transcript — should arrive quickly since
        # Deepgram already received all audio in real-time
        done.wait(timeout=10.0)

    if final_transcript:
        print(f"[DEEPGRAM] {final_transcript}", flush=True)

    return final_transcript
