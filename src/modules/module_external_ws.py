"""
Module: External Server STT (WebSocket streaming + HTTP fallback)

Streams mic audio to the TARS-AI Server's /ws/stt endpoint in real-time
while using local VAD for end-of-speech detection.

Instead of recording the full utterance and then uploading a WAV file
(~1-2s network delay after speech ends), audio frames are sent as they
arrive.  When local VAD signals end-of-speech, we send "end" and the
server already has all the audio — only Whisper inference remains.

If websocket-client is not installed, falls back to HTTP POST to
/transcribe (record-then-upload, slower but zero extra dependencies).

WebSocket protocol:
  1. Open WebSocket to ws://<server>/ws/stt
  2. Send JSON config: {"sample_rate": 16000, "language": "en"}
  3. Send binary frames (int16 PCM) as recorded
  4. Send text "end" when VAD fires
  5. Receive JSON: {"text": "...", "segments": [...], "is_final": true}
"""

import json
import os
import time
import wave
from io import BytesIO

import numpy as np
import requests

from modules.module_messageQue import queue_message
from modules.module_config import load_config

CONFIG = load_config()

# Check for websocket-client once at import time
try:
    import websocket as _websocket
except ImportError:
    _websocket = None


def transcribe_streaming(stt_manager):
    """Stream mic audio to external server.

    Uses WebSocket streaming if websocket-client is installed,
    otherwise falls back to HTTP POST.

    Args:
        stt_manager: The STTManager instance (for VAD, mic, thresholds).

    Returns:
        str: Final transcript text, or None.
    """
    if _websocket is not None:
        return _transcribe_ws(stt_manager)
    return _transcribe_http(stt_manager)


# ---------------------------------------------------------------------------
#  WebSocket streaming path (fast — audio sent while recording)
# ---------------------------------------------------------------------------

def _transcribe_ws(stt_manager):
    from modules.module_mic import ResamplingInputStream
    from modules.module_tts import is_tts_playing, needs_mic_flush, clear_mic_flush
    from modules.module_state import set_tars_state, TarsState

    debug = stt_manager.DEBUG
    external_url = stt_manager.config['STT'].get('external_url', '')
    language = CONFIG['STT'].get('language', '').strip() or None

    # Convert http(s):// to ws(s)://
    ws_url = external_url.replace('https://', 'wss://').replace('http://', 'ws://')
    ws_url = ws_url.rstrip('/') + '/ws/stt'

    t_start = time.monotonic()

    # --- Connect ---
    try:
        ws = _websocket.create_connection(ws_url, timeout=5)
    except Exception as e:
        queue_message(f"ERROR: External WS connection failed: {e}")
        return _transcribe_http(stt_manager)

    t_connected = time.monotonic()
    if debug:
        print(f"[EXTERNAL-WS] Connected in {(t_connected - t_start) * 1000:.0f}ms", flush=True)

    # --- Send config ---
    config_msg = {"sample_rate": stt_manager.MODEL_RATE}
    if language:
        config_msg["language"] = language
    try:
        ws.send(json.dumps(config_msg))
    except Exception as e:
        queue_message(f"ERROR: External WS config send failed: {e}")
        ws.close()
        return _transcribe_http(stt_manager)

    # --- VAD setup (same as module_deepgram) ---
    vad_dispatch = {
        "silero": stt_manager._is_silence_detected_silero,
        "sherpa-onnx": stt_manager._is_silence_detected_sherpa_onnx,
        "smart-turn": stt_manager._is_silence_detected_smart_turn
            if stt_manager.smart_turn_session is not None
            else stt_manager._is_silence_detected_rms,
    }
    vad_func = vad_dispatch.get(stt_manager.vadmethod, stt_manager._is_silence_detected_rms)

    if stt_manager.sherpa_vad is not None:
        stt_manager.sherpa_vad.reset()
    stt_manager.smart_turn_audio_buffer.clear()

    detected_speech = False
    silent_frames = 0
    speech_frames = 0
    max_silent = stt_manager.MAX_SILENT_FRAMES
    min_speech_frames = 5
    aborted = False

    # --- Stream audio ---
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

                # Abort if TTS started playing
                if is_tts_playing():
                    set_tars_state(TarsState.STANDBY)
                    aborted = True
                    break

                # Send amplified audio to server in real-time
                try:
                    ws.send(stt_manager.amplify_audio(data).tobytes(),
                            opcode=_websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    queue_message(f"ERROR: External WS send failed: {e}")
                    break

                # Run local VAD
                is_silence, detected_speech, silent_frames = vad_func(
                    data, detected_speech, silent_frames
                )

                # Pre-speech timeout
                _gesture_running = False
                try:
                    from modules.module_gestures import gesture_active
                    _gesture_running = gesture_active
                except Exception:
                    pass
                if not detected_speech and silent_frames >= max_silent and not _gesture_running:
                    break

                # Post-speech silence timeout
                if is_silence and detected_speech and speech_frames >= min_speech_frames:
                    if silent_frames >= max_silent:
                        if debug:
                            print(f"[EXTERNAL-WS] EOT after {speech_frames} speech frames", flush=True)
                        break

                if detected_speech and not is_silence:
                    speech_frames += 1

    except Exception as e:
        queue_message(f"ERROR: External WS recording error: {e}")

    t_vad_done = time.monotonic()

    # --- No speech: reset and close ---
    if aborted or speech_frames < min_speech_frames:
        try:
            ws.send("reset")
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
        return None

    # --- Signal end-of-audio and wait for transcript ---
    transcript = None
    try:
        ws.send("end")
        ws.settimeout(30)
        response = ws.recv()
        result = json.loads(response)
        transcript = result.get("text", "").strip() or None
        if result.get("error"):
            queue_message(f"ERROR: External WS server error: {result['error']}")
            transcript = None
    except Exception as e:
        queue_message(f"ERROR: External WS receive failed: {e}")
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if debug:
        t_result = time.monotonic()
        if transcript:
            print(f"[EXTERNAL-WS] Final ({(t_result - t_vad_done) * 1000:.0f}ms): {transcript}", flush=True)

    return transcript


# ---------------------------------------------------------------------------
#  HTTP POST fallback (slower — record-then-upload, but no extra deps)
# ---------------------------------------------------------------------------

def _transcribe_http(stt_manager):
    """Record full utterance, then POST to /transcribe endpoint."""
    try:
        chunks, _ = stt_manager._record_audio_chunks()
        if chunks is None:
            return None

        external_url = stt_manager.config['STT'].get('external_url', '')
        language = "en"

        wav_buf = stt_manager._chunks_to_wav_buffer(chunks, stt_manager.MODEL_RATE)
        files = {"audio": ("audio.wav", wav_buf, "audio/wav")}
        data = {"language": language}
        headers = {}
        api_key = os.environ.get('EXTERNAL_API_KEY', '')
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            f"{external_url}/transcribe",
            files=files, data=data, headers=headers, timeout=30
        )
        if response.status_code != 200:
            queue_message(f"ERROR: Server STT returned {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()
        transcript = result.get("text", "").strip()
        if not transcript:
            return None

        return transcript
    except requests.RequestException as e:
        queue_message(f"ERROR: Server transcription request failed: {e}")
    return None
