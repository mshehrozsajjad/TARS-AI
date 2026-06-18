"""
Module: Gladia STT
Real-time speech-to-text using the official Gladia SDK.

Streams mic audio to Gladia in real-time while using local VAD for
end-of-speech detection. Audio is sent as the user speaks, so the
transcript is ready almost instantly after stop_recording() is called.

GladiaClient is reused across sessions to avoid re-initialization.
start_session() is non-blocking (SDK buffers audio until connected),
so per-utterance sessions have negligible overhead.
"""

import os
import threading

from modules.module_messageQue import queue_message


# Reuse client across sessions to avoid re-initialization
_client = None


def transcribe_streaming(stt_manager):
    """Stream mic audio to Gladia while using local VAD for end-of-speech.

    Creates a per-utterance session for fast finalization via stop_recording().
    start_session() is non-blocking — the SDK buffers audio until the
    WebSocket connection is ready, so there's no startup delay.

    Args:
        stt_manager: The STTManager instance (for VAD, mic, thresholds).

    Returns:
        str: Final transcript text, or None.
    """
    global _client

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        print("[GLADIA] ERROR: GLADIA_API_KEY not set", flush=True)
        return None

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
        return None

    from modules.module_mic import ResamplingInputStream
    from modules.module_tts import is_tts_playing, needs_mic_flush, clear_mic_flush
    from modules.module_state import set_tars_state, TarsState

    # Reuse client across sessions
    if _client is None:
        _client = GladiaClient(api_key=api_key)

    live = _client.live()

    final_transcript = None
    done = threading.Event()

    # start_session() is non-blocking — returns immediately, connects
    # in the background, and buffers any audio sent before ready.
    session = live.start_session(
        LiveV2InitRequest(
            model="solaria-3",
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

    @session.on("message")
    def on_message(msg: LiveV2WebSocketMessage):
        nonlocal final_transcript
        if msg.type == "transcript" and msg.data.is_final:
            text = msg.data.utterance.text.strip()
            if text:
                final_transcript = text
                done.set()

    @session.on("error")
    def on_error(err: Exception):
        print(f"[GLADIA] Error: {err}", flush=True)
        done.set()

    @session.once("ended")
    def on_ended(msg: LiveV2EndedMessage):
        done.set()

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

    print(f"[DEBUG] Gladia: starting stream, vad={stt_manager.vadmethod}", flush=True)

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
                session.stop_recording()
                done.wait(timeout=2)
                return None

            # Send audio to Gladia in real-time
            session.send_audio(stt_manager.amplify_audio(data).tobytes())

            # Run local VAD
            is_silence, detected_speech, silent_frames = vad_func(data, detected_speech, silent_frames)

            # Extend pre-speech timeout while a gesture is running —
            # user naturally waits for the gesture to finish before speaking
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
                    print("[GLADIA] Speech detected, streaming...", flush=True)
                speech_frames += 1

    # Signal end of audio — Gladia finalizes immediately
    session.stop_recording()

    if speech_frames < min_speech_frames:
        done.wait(timeout=2)
        return None

    # Wait for final transcript — should arrive very quickly since
    # Gladia already received all audio in real-time and stop_recording
    # triggers immediate finalization
    done.wait(timeout=10.0)

    if final_transcript:
        print(f"[GLADIA] {final_transcript}", flush=True)

    return final_transcript
