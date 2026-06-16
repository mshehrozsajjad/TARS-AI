"""
Module: Gladia STT
Speech-to-text using Gladia's live WebSocket API.

Records audio using the existing VAD pipeline (same as OpenAI STT),
then streams the recorded audio to Gladia for transcription.
This gives us the battle-tested silence detection, amplification,
and noise filtering from the main STT module.

Used as an STT processor option in module_stt.py.
"""

import os
import json
import asyncio
import numpy as np

from modules.module_config import load_config
from modules.module_messageQue import queue_message

CONFIG = load_config()


def transcribe_audio(audio_data, sample_rate=16000):
    """Send recorded audio to Gladia and return the transcript.

    Args:
        audio_data: numpy int16 array of recorded audio (already amplified).
        sample_rate: sample rate of the audio (default 16000).

    Returns:
        str: Transcript text, or None.
    """
    api_key = os.getenv("GLADIA_API_KEY", "")
    if not api_key:
        queue_message("ERROR: GLADIA_API_KEY not set in .env")
        return None

    try:
        import websockets
    except ImportError:
        queue_message("ERROR: pip install websockets")
        return None

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _send_and_transcribe(api_key, websockets, audio_data, sample_rate)
        )
    except Exception as e:
        queue_message(f"ERROR: Gladia STT failed: {e}")
        return None
    finally:
        loop.close()


async def _send_and_transcribe(api_key, websockets_mod, audio_data, sample_rate):
    """Create session, send audio, return transcript."""
    import aiohttp

    # Create session
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
                "sample_rate": sample_rate,
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
                queue_message(f"GLADIA: Session failed ({resp.status})")
                return None
            data = await resp.json()
            ws_url = data.get("url")
            if not ws_url:
                return None

    # Connect and send audio
    final_transcript = None
    done = asyncio.Event()

    ws = await websockets_mod.connect(ws_url)
    try:
        # Receiver
        async def _recv():
            nonlocal final_transcript
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "transcript" and msg.get("data", {}).get("is_final"):
                        text = msg.get("data", {}).get("utterance", {}).get("text", "").strip()
                        if text:
                            final_transcript = text
                            done.set()
                    elif msg.get("type") == "error":
                        queue_message(f"GLADIA: {msg}")
                        done.set()
            except Exception:
                done.set()

        recv_task = asyncio.create_task(_recv())

        # Send audio in chunks (1600 samples = 100ms at 16kHz)
        pcm_bytes = audio_data.tobytes()
        chunk_size = 3200  # 1600 samples * 2 bytes per int16
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            try:
                await ws.send(chunk)
            except Exception:
                break
            # Small delay to simulate real-time pace (prevents overwhelming the API)
            await asyncio.sleep(0.05)

        # Signal end
        try:
            await ws.send(json.dumps({"type": "stop_recording"}))
        except Exception:
            pass

        # Wait for transcript
        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
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
