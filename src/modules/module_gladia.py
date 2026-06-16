"""
Module: Gladia Streaming STT
Real-time speech-to-text using Gladia's WebSocket API.

Streams mic audio to Gladia in real-time and returns the final
transcript when the user stops speaking. Gladia handles VAD
(voice activity detection) internally.

Sessions are pre-created during wake word detection so the WebSocket
is ready to receive audio immediately when recording starts.

Used as an STT processor option in module_stt.py.
"""

import os
import json
import asyncio
import threading
import numpy as np
import time

from modules.module_config import load_config
from modules.module_messageQue import queue_message

CONFIG = load_config()

# ── Lazy imports ─────────────────────────────────────────────────────
_websockets = None


def _ensure_deps():
    global _websockets
    if _websockets is not None:
        return
    try:
        import websockets
        _websockets = websockets
    except ImportError:
        raise ImportError(
            "websockets package is required for Gladia STT. "
            "Install it with: pip install websockets"
        )


# ── Pre-warmed session cache ─────────────────────────────────────────
# Session URL is fetched during wake word so it's ready when recording starts.
_cached_ws_url = None
_cached_session_id = None
_cache_time = 0
_cache_lock = threading.Lock()
_SESSION_TTL = 60  # seconds before cached session expires


def prewarm_session():
    """Create a Gladia session in background. Call during wake word detection."""
    threading.Thread(target=_do_prewarm, daemon=True).start()


def _do_prewarm():
    """Synchronous helper: create session and cache the WebSocket URL."""
    global _cached_ws_url, _cached_session_id, _cache_time
    _ensure_deps()

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        return

    # Skip if we already have a fresh session
    with _cache_lock:
        if _cached_ws_url and (time.time() - _cache_time) < _SESSION_TTL:
            return

    try:
        import requests
        resp = requests.post(
            "https://api.gladia.io/v2/live",
            headers={
                "Content-Type": "application/json",
                "x-gladia-key": api_key,
            },
            json={
                "model": "solaria-1",
                "encoding": "wav/pcm",
                "sample_rate": 16000,
                "bit_depth": 16,
                "channels": 1,
                "language_config": {
                    "languages": ["en"],
                    "code_switching": False,
                },
                "messages_config": {
                    "receive_partial_transcripts": False,
                    "receive_final_transcripts": True,
                },
            },
            timeout=5,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            with _cache_lock:
                _cached_ws_url = data.get("url")
                _cached_session_id = data.get("id")
                _cache_time = time.time()
            queue_message(f"GLADIA: Pre-warmed session {_cached_session_id}")
        else:
            queue_message(f"GLADIA: Pre-warm failed ({resp.status_code})")
    except Exception as e:
        queue_message(f"GLADIA: Pre-warm error: {e}")


def _pop_cached_session():
    """Return and clear the cached session URL, or None if expired/missing."""
    global _cached_ws_url, _cached_session_id, _cache_time
    with _cache_lock:
        if _cached_ws_url and (time.time() - _cache_time) < _SESSION_TTL:
            url = _cached_ws_url
            sid = _cached_session_id
            _cached_ws_url = None
            _cached_session_id = None
            return url, sid
        return None, None


# ── Main transcription function ──────────────────────────────────────

def transcribe_streaming(mic_reader, silence_threshold=None, max_duration=12.5):
    """Stream mic audio to Gladia and return the final transcript.

    Args:
        mic_reader:         A ResamplingInputStream context (already entered).
        silence_threshold:  RMS threshold below which audio is considered silence.
        max_duration:       Max recording duration in seconds.

    Returns:
        str: Final transcript text, or None if nothing was said.
    """
    _ensure_deps()

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        queue_message("ERROR: GLADIA_API_KEY not set in .env")
        return None

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _stream_and_transcribe(api_key, mic_reader, max_duration)
        )
        return result
    except Exception as e:
        queue_message(f"ERROR: Gladia transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        loop.close()


async def _stream_and_transcribe(api_key, mic_reader, max_duration):
    """Core async function: connect to Gladia, stream audio, get transcript."""

    # Try pre-warmed session first
    ws_url, session_id = _pop_cached_session()

    # Fall back to creating a new session
    if not ws_url:
        import aiohttp
        async with aiohttp.ClientSession() as http:
            async with http.post(
                "https://api.gladia.io/v2/live",
                headers={
                    "Content-Type": "application/json",
                    "x-gladia-key": api_key,
                },
                json={
                    "model": "solaria-1",
                    "encoding": "wav/pcm",
                    "sample_rate": 16000,
                    "bit_depth": 16,
                    "channels": 1,
                    "language_config": {
                        "languages": ["en"],
                        "code_switching": False,
                    },
                    "messages_config": {
                        "receive_partial_transcripts": False,
                        "receive_final_transcripts": True,
                    },
                },
            ) as resp:
                if resp.status not in (200, 201):
                    error = await resp.text()
                    queue_message(f"ERROR: Gladia session creation failed ({resp.status}): {error}")
                    return None
                data = await resp.json()
                ws_url = data.get("url")
                session_id = data.get("id")
                if not ws_url:
                    queue_message("ERROR: Gladia returned no WebSocket URL")
                    return None
                queue_message(f"GLADIA: Session {session_id} created")
    else:
        queue_message(f"GLADIA: Using pre-warmed session {session_id}")

    # Connect WebSocket and stream audio
    final_transcript = None
    transcript_event = asyncio.Event()

    try:
        ws = await _websockets.connect(ws_url)
    except Exception as e:
        queue_message(f"GLADIA: WebSocket connect failed (session may have expired): {e}")
        return None

    try:
        async def _receive():
            nonlocal final_transcript
            try:
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    if msg_type == "transcript" and msg.get("data", {}).get("is_final"):
                        text = msg["data"].get("utterance", {}).get("text", "").strip()
                        if text:
                            final_transcript = text
                            queue_message(f"GLADIA: Final transcript: {text}")
                            transcript_event.set()

                    elif msg_type == "post_final_transcript":
                        transcript_event.set()

                    elif msg_type == "error":
                        queue_message(f"GLADIA: Error: {msg}")
                        transcript_event.set()

            except Exception as e:
                queue_message(f"GLADIA: Receive error: {e}")
                transcript_event.set()

        recv_task = asyncio.create_task(_receive())

        # Stream mic audio
        CHUNK_FRAMES = 1600  # 100ms at 16kHz
        loop = asyncio.get_event_loop()
        max_chunks = int(max_duration * 10)
        chunks_sent = 0
        silence_count = 0
        speech_detected = False
        MAX_SILENCE_CHUNKS = 15  # 1.5s of silence after speech = done

        for _ in range(max_chunks):
            if transcript_event.is_set():
                break

            def _read():
                data, _ = mic_reader.read(CHUNK_FRAMES)
                if data.dtype != np.int16:
                    data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
                return data

            chunk = await loop.run_in_executor(None, _read)
            pcm_bytes = chunk.tobytes()

            await ws.send(pcm_bytes)
            chunks_sent += 1

            # Simple silence detection
            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            if rms > 500:
                speech_detected = True
                silence_count = 0
            elif speech_detected:
                silence_count += 1
                if silence_count >= MAX_SILENCE_CHUNKS:
                    break

        # Signal end of audio
        try:
            await ws.send(json.dumps({"type": "stop_recording"}))
        except Exception:
            pass

        # Wait for final transcript
        try:
            await asyncio.wait_for(transcript_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            queue_message("GLADIA: Timeout waiting for final transcript")

        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    finally:
        try:
            await ws.close()
        except Exception:
            pass

    queue_message(f"GLADIA: Sent {chunks_sent} audio chunks")
    return final_transcript
