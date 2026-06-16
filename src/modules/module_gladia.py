"""
Module: Gladia Streaming STT
Real-time speech-to-text using Gladia's WebSocket API.

Maintains a persistent WebSocket session that's reused across turns.
Session is created once and kept alive — no per-turn creation overhead.

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


class GladiaSession:
    """Persistent Gladia live transcription session.

    Runs its own asyncio event loop in a background thread.
    The WebSocket stays connected between turns.
    """

    def __init__(self):
        self._api_key = os.getenv("GLADIA_API_KEY", "")
        self._ws = None
        self._loop = None
        self._thread = None
        self._running = False

        # Communication between main thread and async loop
        self._audio_queue = None       # main → async: audio bytes to send
        self._transcript_final = None  # async → main: final transcript text
        self._transcript_partial = None
        self._done = threading.Event()
        self._turn_active = threading.Event()
        self._on_partial = None

    def ensure_connected(self):
        """Start the background thread and connect if not already running."""
        if self._running and self._thread and self._thread.is_alive():
            return True

        if not self._api_key:
            queue_message("ERROR: GLADIA_API_KEY not set")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for connection
        for _ in range(50):  # 5 seconds max
            if self._ws_connected:
                return True
            time.sleep(0.1)

        queue_message("GLADIA: Connection timeout")
        return False

    @property
    def _ws_connected(self):
        return self._ws is not None

    def _run_loop(self):
        """Background thread: runs the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_loop())
        except Exception as e:
            queue_message(f"GLADIA: Session loop error: {e}")
        finally:
            self._running = False
            self._loop.close()

    async def _session_loop(self):
        """Main async loop: connect and handle turns until stopped."""
        try:
            import websockets
        except ImportError:
            queue_message("ERROR: pip install websockets")
            return

        while self._running:
            # Create session
            ws_url = await self._create_session()
            if not ws_url:
                await asyncio.sleep(2)
                continue

            # Connect WebSocket
            try:
                self._ws = await websockets.connect(ws_url)
                queue_message("GLADIA: Session connected")
            except Exception as e:
                queue_message(f"GLADIA: WebSocket connect failed: {e}")
                self._ws = None
                await asyncio.sleep(2)
                continue

            # Handle turns on this connection
            try:
                await self._handle_connection()
            except Exception as e:
                queue_message(f"GLADIA: Connection error: {e}")
            finally:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

            if not self._running:
                break
            queue_message("GLADIA: Reconnecting...")

    async def _create_session(self):
        """Create a Gladia live session, return WebSocket URL."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as http:
                async with http.post(
                    "https://api.gladia.io/v2/live",
                    headers={
                        "Content-Type": "application/json",
                        "x-gladia-key": self._api_key,
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
                        queue_message(f"GLADIA: Session creation failed ({resp.status})")
                        return None
                    data = await resp.json()
                    queue_message("GLADIA: Session ready")
                    return data.get("url")
        except Exception as e:
            queue_message(f"GLADIA: Session creation error: {e}")
            return None

    async def _handle_connection(self):
        """Handle the WebSocket connection — receive transcripts, wait for turns."""
        recv_task = asyncio.create_task(self._receive_loop())
        try:
            # Stay alive until connection drops or we're stopped
            while self._running:
                # Wait for a turn to start
                while self._running and not self._turn_active.is_set():
                    await asyncio.sleep(0.05)

                if not self._running:
                    break

                # Process audio for this turn
                await self._process_turn()

        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    async def _receive_loop(self):
        """Continuously receive messages from Gladia."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "transcript":
                    text = msg.get("data", {}).get("utterance", {}).get("text", "").strip()
                    if not text:
                        continue
                    if msg["data"].get("is_final"):
                        self._transcript_final = text
                        self._done.set()
                    else:
                        self._transcript_partial = text
                        if self._on_partial:
                            try:
                                self._on_partial(text)
                            except Exception:
                                pass
                elif msg.get("type") == "error":
                    queue_message(f"GLADIA: {msg}")
                    self._done.set()
        except Exception:
            self._done.set()
            raise  # Let _handle_connection catch it and reconnect

    async def _process_turn(self):
        """Send audio from queue until turn ends."""
        while self._turn_active.is_set() and self._running:
            try:
                audio_bytes = self._audio_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.01)
                continue

            if audio_bytes is None:  # turn end signal
                break

            try:
                await self._ws.send(audio_bytes)
            except Exception:
                break

    def transcribe(self, mic_reader, on_partial=None, max_duration=12.5):
        """Stream mic audio and return the final transcript. Blocks until done."""
        if not self.ensure_connected():
            return None

        # Reset state for this turn
        self._transcript_final = None
        self._transcript_partial = None
        self._done.clear()
        self._on_partial = on_partial
        self._audio_queue = __import__('queue').Queue()

        # Signal turn start
        self._turn_active.set()

        # Stream mic audio from this thread
        CHUNK = 1600  # 100ms at 16kHz
        sent = 0
        silence = 0
        heard_speech = False

        try:
            for _ in range(int(max_duration * 10)):
                if self._done.is_set():
                    break

                data, _ = mic_reader.read(CHUNK)
                if data.dtype != np.int16:
                    data = np.clip(data * 32768, -32768, 32767).astype(np.int16)

                self._audio_queue.put(data.tobytes())
                sent += 1

                rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
                if rms > 500:
                    heard_speech = True
                    silence = 0
                elif heard_speech:
                    silence += 1
                    if silence >= 15:  # 1.5s silence
                        break
        finally:
            self._audio_queue.put(None)  # end signal
            self._turn_active.clear()

        # Wait for final transcript
        self._done.wait(timeout=5.0)

        result = self._transcript_final
        if result:
            queue_message(f"GLADIA: {result}")

        self._on_partial = None
        return result

    def close(self):
        """Shut down the session."""
        self._running = False
        self._turn_active.set()  # unblock any waiting
        self._done.set()
        if self._thread:
            self._thread.join(timeout=5)


# ── Module-level singleton ───────────────────────────────────────────

_session = None


def get_session():
    global _session
    if _session is None:
        _session = GladiaSession()
    return _session


def transcribe_streaming(mic_reader, on_partial=None, max_duration=12.5):
    """Main entry point called from module_stt.py."""
    return get_session().transcribe(mic_reader, on_partial, max_duration)
