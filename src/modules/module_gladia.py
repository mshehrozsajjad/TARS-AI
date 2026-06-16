"""
Module: Gladia Streaming STT
Real-time speech-to-text using Gladia's WebSocket API.

Streams mic audio to Gladia in real-time and returns the final
transcript when the user stops speaking. Gladia handles VAD
(voice activity detection) internally.

Used as an STT processor option in module_stt.py.
"""

import os
import json
import asyncio
import threading
import numpy as np

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
    """Core async function: open Gladia session, stream audio, get transcript."""
    import aiohttp

    # Step 1: Create a live session via REST to get WebSocket URL
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
            if resp.status != 200:
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

    # Step 2: Connect WebSocket and stream audio
    final_transcript = None
    transcript_event = asyncio.Event()

    async with _websockets.connect(ws_url) as ws:
        # Background task: receive transcripts
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
                        # Session is wrapping up
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
        max_chunks = int(max_duration * 10)  # 10 chunks per second
        chunks_sent = 0
        silence_count = 0
        speech_detected = False
        MAX_SILENCE_CHUNKS = 15  # 1.5s of silence after speech = done

        for _ in range(max_chunks):
            if transcript_event.is_set():
                break

            # Read mic chunk (blocking, run in executor)
            def _read():
                data, _ = mic_reader.read(CHUNK_FRAMES)
                if data.dtype != np.int16:
                    data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
                return data

            chunk = await loop.run_in_executor(None, _read)
            pcm_bytes = chunk.tobytes()

            # Send as binary frame
            await ws.send(pcm_bytes)
            chunks_sent += 1

            # Simple silence detection to know when to stop
            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            if rms > 500:  # speech detected
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

        # Wait for final transcript (up to 5 seconds)
        try:
            await asyncio.wait_for(transcript_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            queue_message("GLADIA: Timeout waiting for final transcript")

        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    queue_message(f"GLADIA: Sent {chunks_sent} audio chunks")
    return final_transcript
