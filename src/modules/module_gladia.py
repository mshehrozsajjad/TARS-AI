"""
Module: Gladia Streaming STT
Real-time speech-to-text using Gladia's WebSocket API.

Streams mic audio to Gladia in real-time and returns the final
transcript when the user stops speaking. Shows partial transcripts
in real-time on the UI as the user speaks.

Session is created once during wake word detection and reused
across turns to avoid per-turn creation delay.

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


# ── Persistent session ──────────────────────────────────────────────
# Created during wake word, reused across turns until it drops.

_session_ws = None
_session_url = None
_session_id = None
_session_lock = threading.Lock()
_session_ready = threading.Event()


def prepare_session():
    """Create a Gladia session in background. Call from wake_word_callback."""
    _ensure_deps()
    threading.Thread(target=_create_session_sync, daemon=True).start()


def _create_session_sync():
    """Create session via HTTP and connect WebSocket."""
    global _session_ws, _session_url, _session_id

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        return

    # Skip if we already have a live session
    with _session_lock:
        if _session_ws is not None:
            try:
                # Check if still open
                if _session_ws.open:
                    _session_ready.set()
                    return
            except Exception:
                pass
            _session_ws = None

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
                "language_config": {"languages": ["en"], "code_switching": False},
                "messages_config": {
                    "receive_partial_transcripts": True,
                    "receive_final_transcripts": True,
                },
            },
            timeout=5,
        )
        if resp.status_code not in (200, 201):
            queue_message(f"GLADIA: Session creation failed ({resp.status_code})")
            return

        data = resp.json()
        with _session_lock:
            _session_url = data.get("url")
            _session_id = data.get("id")
        queue_message(f"GLADIA: Session {_session_id} ready")
        _session_ready.set()

    except Exception as e:
        queue_message(f"GLADIA: Session creation error: {e}")


def _get_or_create_session(api_key):
    """Get the pre-created session URL or create one on the spot."""
    global _session_url, _session_id

    # Wait briefly for pre-created session
    if _session_ready.wait(timeout=3.0):
        with _session_lock:
            if _session_url:
                url, sid = _session_url, _session_id
                # Don't clear — reuse for next turn by creating a new one
                return url, sid

    # Fallback: create synchronously
    queue_message("GLADIA: Creating session (no pre-warm available)")
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_create_session_async(api_key))
    finally:
        loop.close()


async def _create_session_async(api_key):
    """Create session via async HTTP."""
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
                "language_config": {"languages": ["en"], "code_switching": False},
                "messages_config": {
                    "receive_partial_transcripts": True,
                    "receive_final_transcripts": True,
                },
            },
        ) as resp:
            if resp.status not in (200, 201):
                return None, None
            data = await resp.json()
            return data.get("url"), data.get("id")


# ── Main transcription function ──────────────────────────────────────

def transcribe_streaming(mic_reader, on_partial=None, max_duration=12.5):
    """Stream mic audio to Gladia and return the final transcript.

    Args:
        mic_reader:    ResamplingInputStream context (already entered).
        on_partial:    Optional callback(text) for partial transcripts (UI display).
        max_duration:  Max recording duration in seconds.

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
        return loop.run_until_complete(
            _stream_and_transcribe(api_key, mic_reader, on_partial, max_duration)
        )
    except Exception as e:
        queue_message(f"ERROR: Gladia transcription failed: {e}")
        return None
    finally:
        loop.close()


async def _stream_and_transcribe(api_key, mic_reader, on_partial, max_duration):
    """Connect to Gladia, stream audio, show partials, return final transcript."""
    global _session_url, _session_id

    # Get session URL (pre-created or on-demand)
    ws_url, session_id = _get_or_create_session(api_key)
    if not ws_url:
        queue_message("GLADIA: No session URL available")
        return None

    # Reset for next turn
    _session_ready.clear()
    with _session_lock:
        _session_url = None
        _session_id = None

    final_transcript = None
    transcript_event = asyncio.Event()

    try:
        ws = await _websockets.connect(ws_url)
    except Exception as e:
        queue_message(f"GLADIA: WebSocket connect failed: {e}")
        return None

    try:
        # Background receiver
        async def _receive():
            nonlocal final_transcript
            try:
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    if msg_type == "transcript":
                        is_final = msg.get("data", {}).get("is_final", False)
                        text = msg.get("data", {}).get("utterance", {}).get("text", "").strip()

                        if text and is_final:
                            final_transcript = text
                            queue_message(f"GLADIA: {text}")
                            transcript_event.set()
                        elif text and on_partial:
                            try:
                                on_partial(text)
                            except Exception:
                                pass

                    elif msg_type == "error":
                        queue_message(f"GLADIA: Error: {msg}")
                        transcript_event.set()
            except Exception:
                transcript_event.set()

        recv_task = asyncio.create_task(_receive())

        # Stream mic audio
        CHUNK_FRAMES = 1600  # 100ms at 16kHz
        eloop = asyncio.get_event_loop()
        max_chunks = int(max_duration * 10)
        chunks_sent = 0
        silence_count = 0
        speech_detected = False
        MAX_SILENCE_CHUNKS = 15  # 1.5s silence = done

        for _ in range(max_chunks):
            if transcript_event.is_set():
                break

            def _read():
                data, _ = mic_reader.read(CHUNK_FRAMES)
                if data.dtype != np.int16:
                    data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
                return data

            chunk = await eloop.run_in_executor(None, _read)

            try:
                await ws.send(chunk.tobytes())
                chunks_sent += 1
            except Exception:
                break

            # Silence detection
            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            if rms > 500:
                speech_detected = True
                silence_count = 0
            elif speech_detected:
                silence_count += 1
                if silence_count >= MAX_SILENCE_CHUNKS:
                    break

        # Signal end
        try:
            await ws.send(json.dumps({"type": "stop_recording"}))
        except Exception:
            pass

        # Wait for final transcript
        try:
            await asyncio.wait_for(transcript_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            queue_message("GLADIA: Timeout waiting for transcript")

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

    return final_transcript
