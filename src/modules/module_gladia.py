"""
Module: Gladia Streaming STT
Real-time speech-to-text using Gladia's WebSocket API.

Streams mic audio to Gladia in real-time and returns the final
transcript when the user stops speaking.

Used as an STT processor option in module_stt.py.
"""

import os
import json
import asyncio
import numpy as np

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


def transcribe_streaming(mic_reader, max_duration=12.5):
    """Stream mic audio to Gladia and return the final transcript."""
    _ensure_deps()

    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        queue_message("ERROR: GLADIA_API_KEY not set in .env")
        return None

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _stream_and_transcribe(api_key, mic_reader, max_duration)
        )
    except Exception as e:
        queue_message(f"ERROR: Gladia transcription failed: {e}")
        return None
    finally:
        loop.close()


async def _create_session(api_key):
    """Create a Gladia live session. Returns (ws_url, session_id) or (None, None)."""
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
                    "receive_partial_transcripts": False,
                    "receive_final_transcripts": True,
                },
            },
        ) as resp:
            if resp.status not in (200, 201):
                error = await resp.text()
                queue_message(f"ERROR: Gladia session failed ({resp.status}): {error}")
                return None, None
            data = await resp.json()
            return data.get("url"), data.get("id")


async def _stream_and_transcribe(api_key, mic_reader, max_duration):
    """Create session, connect WebSocket, stream audio, return transcript."""

    ws_url, session_id = await _create_session(api_key)
    if not ws_url:
        return None
    queue_message(f"GLADIA: Session {session_id} created")

    final_transcript = None
    transcript_event = asyncio.Event()

    try:
        ws = await _websockets.connect(ws_url)
    except Exception as e:
        queue_message(f"GLADIA: WebSocket connect failed: {e}")
        return None

    try:
        # Background receiver for transcripts
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
                    elif msg_type == "error":
                        queue_message(f"GLADIA: Error: {msg}")
                        transcript_event.set()
            except Exception:
                transcript_event.set()

        recv_task = asyncio.create_task(_receive())

        # Stream mic audio
        CHUNK_FRAMES = 1600  # 100ms at 16kHz
        loop = asyncio.get_event_loop()
        max_chunks = int(max_duration * 10)
        chunks_sent = 0
        silence_count = 0
        speech_detected = False
        MAX_SILENCE_CHUNKS = 15  # 1.5s silence after speech = done

        for _ in range(max_chunks):
            if transcript_event.is_set():
                break

            def _read():
                data, _ = mic_reader.read(CHUNK_FRAMES)
                if data.dtype != np.int16:
                    data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
                return data

            chunk = await loop.run_in_executor(None, _read)

            try:
                await ws.send(chunk.tobytes())
                chunks_sent += 1
            except Exception:
                break  # connection dropped

            # Silence detection
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

    queue_message(f"GLADIA: Sent {chunks_sent} chunks")
    return final_transcript
