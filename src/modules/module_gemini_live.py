"""
Module: Gemini Live
Manages a Gemini Live API session for real-time audio conversation.

Replaces the entire STT → LLM → TTS pipeline with direct audio-in,
audio-out streaming via Google's Gemini Live API.

Audio is captured from the shared mic hub, streamed over a persistent
WebSocket to Gemini, and audio responses are played directly through
the speaker — Gemini handles STT, LLM, and TTS in one round trip.

The module exposes a single entry point — run_gemini_live_turn() —
called from module_main.py after wake word detection.
"""

import os
import asyncio
import threading
import numpy as np
import sounddevice as sd

from modules.module_config import load_config, get_capabilities
from modules.module_messageQue import queue_message

CONFIG = load_config()

# ── Lazy SDK import ──────────────────────────────────────────────────
_genai = None
_types = None


def _ensure_sdk():
    """Import the google-genai SDK on first use."""
    global _genai, _types
    if _genai is not None:
        return
    try:
        from google import genai
        from google.genai import types
        _genai = genai
        _types = types
    except ImportError:
        raise ImportError(
            "google-genai package is required for Gemini Live mode. "
            "Install it with: pip install google-genai"
        )


# ── Audio output helpers ─────────────────────────────────────────────

def _get_output_device():
    """Reuse the TTS module's resolved output device."""
    try:
        from modules.module_tts import _resolve_output_device, _output_device
        _resolve_output_device()
        return _output_device
    except Exception:
        return None


# ── Gemini Live Session Manager ──────────────────────────────────────

class GeminiLiveSession:
    """Manages a single Gemini Live API conversation session.

    Uses AUDIO response modality — Gemini returns raw PCM audio that
    is played directly through the speaker.  No separate TTS needed.
    """

    GEMINI_SAMPLE_RATE = 24000  # Gemini outputs 24kHz PCM
    PLAYBACK_RATE = 16000       # Resample to 16kHz for USB audio devices

    def __init__(self):
        _ensure_sdk()

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini Live mode. "
                "Set it in your .env file."
            )

        self._client = _genai.Client(api_key=api_key)
        self._session = None
        self._connection_manager = None
        self._connected = False

        # Config
        gemini_cfg = CONFIG.get("GEMINI_LIVE", {})
        self._model = gemini_cfg.get("model", "gemini-3.1-flash-live-preview")
        self._temperature = float(gemini_cfg.get("temperature", CONFIG["LLM"].get("temperature", 0.8)))
        self._vad_silence_ms = int(gemini_cfg.get("vad_silence_ms", 700))
        self._vad_prefix_ms = int(gemini_cfg.get("vad_prefix_ms", 200))
        self._context_trigger_tokens = int(gemini_cfg.get("context_trigger_tokens", 32000))

    async def connect(self, system_prompt=None):
        """Open a persistent Gemini Live session with AUDIO output."""
        if self._connected:
            return

        if system_prompt is None:
            system_prompt = self._build_system_prompt()

        tools_config = self._build_tools()

        config = _types.LiveConnectConfig(
            response_modalities=[_types.Modality.AUDIO],
            system_instruction=system_prompt,
            context_window_compression=_types.ContextWindowCompressionConfig(
                sliding_window=_types.SlidingWindow(),
                trigger_tokens=self._context_trigger_tokens,
            ),
            input_audio_transcription=_types.AudioTranscriptionConfig(),
            output_audio_transcription=_types.AudioTranscriptionConfig(),
            realtime_input_config=_types.RealtimeInputConfig(
                activity_handling=_types.ActivityHandling.NO_INTERRUPTION,
                automatic_activity_detection=_types.AutomaticActivityDetection(
                    disabled=False,
                    prefix_padding_ms=self._vad_prefix_ms,
                    silence_duration_ms=self._vad_silence_ms,
                    start_of_speech_sensitivity=_types.StartSensitivity.START_SENSITIVITY_HIGH,
                ),
            ),
            generation_config=_types.GenerationConfig(
                temperature=self._temperature,
            ),
        )

        if tools_config:
            config.tools = tools_config

        self._connection_manager = self._client.aio.live.connect(
            model=self._model,
            config=config,
        )
        self._session = await self._connection_manager.__aenter__()
        self._connected = True
        queue_message(f"GEMINI_LIVE: Connected to {self._model} (AUDIO mode)")

    async def close(self):
        """Close the Gemini Live session."""
        if not self._connected:
            return
        try:
            if self._connection_manager:
                await self._connection_manager.__aexit__(None, None, None)
        except Exception as e:
            queue_message(f"GEMINI_LIVE: Error closing session: {e}")
        finally:
            self._session = None
            self._connection_manager = None
            self._connected = False
            queue_message("GEMINI_LIVE: Session closed")

    async def run_turn(self, on_audio_chunk=None, on_input_transcript=None,
                       on_output_transcript=None, stop_event=None):
        """Stream mic audio to Gemini and play audio response directly.

        Args:
            on_audio_chunk:       Optional callback(pcm_bytes) for each audio chunk.
            on_input_transcript:  Optional callback(text) when Gemini transcribes
                                  user speech.
            on_output_transcript: Optional callback(text) when Gemini transcribes
                                  its own audio response.
            stop_event:           Optional threading.Event — if set, abort the turn.

        Returns:
            dict with:
                'input_transcript':  User's speech transcription
                'output_transcript': Gemini's response transcription
                'tool_calls':        List of tool call results (if any)
        """
        if not self._connected:
            await self.connect()

        input_transcript_parts = []
        output_transcript_parts = []
        tool_results = []
        audio_chunks = []

        # Open audio output stream at 16kHz (supported by USB audio devices)
        output_device = _get_output_device()
        audio_stream = sd.OutputStream(
            samplerate=self.PLAYBACK_RATE,
            channels=1,
            dtype='int16',
            blocksize=4096,
            device=output_device,
        )
        audio_stream.start()

        # Start mic audio sender in background
        audio_send_task = asyncio.create_task(self._stream_audio(stop_event))

        try:
            async for msg in self._session.receive():
                if stop_event and stop_event.is_set():
                    break

                # Audio response from Gemini — play directly
                if hasattr(msg, 'server_content') and msg.server_content:
                    sc = msg.server_content

                    # Audio data in model turn
                    if hasattr(sc, 'model_turn') and sc.model_turn:
                        for part in sc.model_turn.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                pcm_data = part.inline_data.data
                                if pcm_data:
                                    # Convert 24kHz PCM to int16 numpy array
                                    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
                                    # Resample from 24kHz to 16kHz via linear interpolation
                                    ratio = self.PLAYBACK_RATE / self.GEMINI_SAMPLE_RATE
                                    new_len = int(len(samples) * ratio)
                                    indices = np.linspace(0, len(samples) - 1, new_len)
                                    resampled = np.interp(indices, np.arange(len(samples)), samples)
                                    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
                                    audio_stream.write(resampled.reshape(-1, 1))
                                    audio_chunks.append(pcm_data)
                                    if on_audio_chunk:
                                        try:
                                            on_audio_chunk(pcm_data)
                                        except Exception:
                                            pass

                    # User speech transcription
                    if hasattr(sc, 'input_transcription') and sc.input_transcription:
                        t = sc.input_transcription
                        if hasattr(t, 'text') and t.text:
                            input_transcript_parts.append(t.text)
                            if on_input_transcript:
                                try:
                                    on_input_transcript(t.text)
                                except Exception:
                                    pass

                    # Gemini's own response transcription
                    if hasattr(sc, 'output_transcription') and sc.output_transcription:
                        t = sc.output_transcription
                        if hasattr(t, 'text') and t.text:
                            output_transcript_parts.append(t.text)
                            if on_output_transcript:
                                try:
                                    on_output_transcript(t.text)
                                except Exception:
                                    pass

                    # Turn complete
                    if hasattr(sc, 'turn_complete') and sc.turn_complete:
                        break

                # Tool/function calls
                if hasattr(msg, 'tool_call') and msg.tool_call:
                    result = await self._handle_tool_call(msg.tool_call)
                    if result:
                        tool_results.append(result)

        except Exception as e:
            queue_message(f"GEMINI_LIVE: Error during turn: {e}")
        finally:
            # Stop mic streaming
            audio_send_task.cancel()
            try:
                await audio_send_task
            except asyncio.CancelledError:
                pass
            # Close audio output
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception:
                pass

        input_text = ' '.join(input_transcript_parts)
        output_text = ' '.join(output_transcript_parts)

        if input_text:
            queue_message(f"GEMINI_LIVE: User said: {input_text[:120]}")
        if output_text:
            queue_message(f"GEMINI_LIVE: Gemini said: {output_text[:120]}")

        return {
            'input_transcript': input_text,
            'output_transcript': output_text,
            'tool_calls': tool_results,
        }

    # ── Audio input streaming ────────────────────────────────────

    async def _stream_audio(self, stop_event=None):
        """Read from shared mic hub and stream PCM to Gemini."""
        from modules.module_mic import ResamplingInputStream

        CHUNK_FRAMES = 1600  # 100ms at 16kHz
        AUDIO_MIME = "audio/pcm;rate=16000"
        loop = asyncio.get_event_loop()

        def _read_mic_chunk(mic):
            data, _ = mic.read(CHUNK_FRAMES)
            if data.dtype != np.int16:
                data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
            return data.tobytes()

        with ResamplingInputStream(dtype="int16") as mic:
            mic.flush()
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    audio_bytes = await loop.run_in_executor(None, _read_mic_chunk, mic)
                    await self._session.send_realtime_input(
                        audio=_types.Blob(data=audio_bytes, mime_type=AUDIO_MIME)
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    queue_message(f"GEMINI_LIVE: Audio send error: {e}")
                    break

    # ── Tool/function calling ────────────────────────────────────

    def _build_tools(self):
        """Convert TARS skills to Gemini FunctionDeclaration format."""
        try:
            from modules.module_skills import get_skill_manager
            skills = get_skill_manager()
            if not skills:
                return None

            declarations = []
            for name in skills.get_skill_names():
                meta = skills._skill_meta.get(name, {})
                req_params = meta.get("required_params", [])
                description = meta.get("description", f"Execute the {name} skill")

                properties = {}
                required = []
                for param in req_params:
                    properties[param] = {
                        "type": "string",
                        "description": f"The {param} parameter",
                    }
                    required.append(param)

                params_schema = {"type": "object", "properties": properties}
                if required:
                    params_schema["required"] = required

                declarations.append(
                    _types.FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters_json_schema=params_schema,
                    )
                )

            if declarations:
                return [_types.Tool(function_declarations=declarations)]
            return None
        except Exception as e:
            queue_message(f"GEMINI_LIVE: Failed to build tools: {e}")
            return None

    async def _handle_tool_call(self, tool_call):
        """Execute a Gemini tool call through the TARS skill system."""
        try:
            from modules.module_skills import get_skill_manager
            skills = get_skill_manager()

            results = []
            for fc in tool_call.function_calls:
                func_name = fc.name
                parameters = dict(fc.args) if fc.args else {}
                queue_message(f"GEMINI_LIVE: Tool call: {func_name}({parameters})")

                if skills and skills.has_skill(func_name):
                    context = {
                        "bot_response": {},
                        "user_input": "",
                        "source": "voice",
                        "has_image": False,
                        "config": CONFIG,
                    }
                    result = skills.execute(func_name, parameters, context)
                    result_str = str(result) if result else "Done"
                else:
                    result_str = f"Unknown skill: {func_name}"

                results.append({"name": func_name, "result": result_str})

                response = _types.FunctionResponse(
                    name=func_name,
                    response={"result": result_str},
                    id=fc.id,
                )
                await self._session.send_tool_response(function_responses=response)

            return results
        except Exception as e:
            queue_message(f"GEMINI_LIVE: Tool call error: {e}")
            return None

    # ── System prompt ────────────────────────────────────────────

    def _build_system_prompt(self):
        """Build a system prompt from the character config."""
        parts = []

        sys_prompt = CONFIG['LLM'].get('systemprompt', '')
        if sys_prompt:
            parts.append(sys_prompt)

        try:
            from modules.module_llm import character_manager
            if character_manager:
                char_data = character_manager.get_character_data()
                if char_data:
                    if char_data.get('description'):
                        parts.append(f"Character description: {char_data['description']}")
                    if char_data.get('personality'):
                        parts.append(f"Personality: {char_data['personality']}")
                    if char_data.get('scenario'):
                        parts.append(f"Scenario: {char_data['scenario']}")
        except Exception:
            pass

        user_name = CONFIG['CHAR'].get('user_name', 'User')
        parts.append(f"The user's name is {user_name}.")

        parts.append(
            "Respond naturally in conversation. Keep responses concise and conversational."
        )

        return "\n\n".join(parts)

    @property
    def is_connected(self):
        return self._connected


# ── Module-level session management ──────────────────────────────────

_session_instance = None
_session_lock = threading.Lock()


def get_session():
    """Get or create the global GeminiLiveSession."""
    global _session_instance
    with _session_lock:
        if _session_instance is None:
            _session_instance = GeminiLiveSession()
        return _session_instance


def close_session():
    """Close the global session (call on shutdown)."""
    global _session_instance
    with _session_lock:
        if _session_instance is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_session_instance.close())
                loop.close()
            except Exception as e:
                queue_message(f"GEMINI_LIVE: Error closing global session: {e}")
            _session_instance = None


def run_gemini_live_turn(on_input_transcript=None, on_output_transcript=None,
                         stop_event=None):
    """Synchronous wrapper: run one conversation turn with Gemini Live.

    Main entry point called from module_main.py after wake word.
    Streams mic audio to Gemini, plays Gemini's audio response directly.

    Returns:
        dict with 'input_transcript', 'output_transcript', 'tool_calls'
        or None on error.
    """
    try:
        session = get_session()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                _run_turn_async(session, on_input_transcript,
                                on_output_transcript, stop_event)
            )
            return result
        finally:
            loop.close()
    except Exception as e:
        queue_message(f"GEMINI_LIVE: Turn failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _run_turn_async(session, on_input_transcript, on_output_transcript,
                          stop_event):
    """Async helper: connect if needed, then run a turn."""
    if not session.is_connected:
        await session.connect()
    return await session.run_turn(
        on_input_transcript=on_input_transcript,
        on_output_transcript=on_output_transcript,
        stop_event=stop_event,
    )
