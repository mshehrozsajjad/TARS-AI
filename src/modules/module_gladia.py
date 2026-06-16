"""
Module: Gladia STT
Speech-to-text using the official Gladia SDK.

Uses the sync SDK client which manages HTTP session creation
and WebSocket connection internally.

Used as an STT processor option in module_stt.py.
"""

import os
import threading
import numpy as np

from modules.module_messageQue import queue_message


def transcribe_audio(audio_data, sample_rate=16000):
    """Send recorded audio to Gladia and return the transcript.

    Args:
        audio_data: numpy int16 array of recorded audio.
        sample_rate: sample rate (default 16000).

    Returns:
        str: Transcript text, or None.
    """
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
            LiveV2InitResponse,
        )
    except ImportError:
        print("[GLADIA] ERROR: pip install gladiaio-sdk", flush=True)
        return None

    client = GladiaClient(api_key=api_key)
    live = client.live()

    final_transcript = None
    done = threading.Event()

    session = live.start_session(
        LiveV2InitRequest(
            model="solaria-1",
            encoding="wav/pcm",
            sample_rate=sample_rate,
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

    # Send audio in chunks
    pcm_bytes = audio_data.tobytes()
    chunk_size = 3200  # 100ms at 16kHz (1600 samples * 2 bytes)
    for i in range(0, len(pcm_bytes), chunk_size):
        session.send_audio(pcm_bytes[i:i + chunk_size])

    session.stop_recording()

    # Wait for final transcript
    done.wait(timeout=10.0)

    if final_transcript:
        print(f"[GLADIA] {final_transcript}", flush=True)

    return final_transcript
