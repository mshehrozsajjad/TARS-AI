"""
TARS LiveKit Agent — runs on LiveKit Cloud or a server.

Joins the same LiveKit room as the Pi client. Subscribes to Pi's mic audio,
processes it through STT → LLM → TTS, and publishes audio + video back.
Uses @function_tool for skills — physical actions are forwarded to the Pi
via RPC, virtual actions (web search, etc.) run directly on the agent.

Video avatar powered by Beyond Presence (Bey) — publishes a video track
that the Pi client receives and renders on its Pygame display.

Run:
    python agent.py dev          # local development
    python agent.py start        # production
"""

import asyncio
import logging
import os
import json
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    AutoSubscribe,
    RunContext,
    function_tool,
    cli,
)
from livekit.plugins import bey, deepgram, elevenlabs, openai

load_dotenv()

logger = logging.getLogger("tars-agent")

# ── Config from env ──────────────────────────────────────────────────

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BEY_AVATAR_ENABLED = os.getenv("BEY_AVATAR_ENABLED", "true").lower() not in ("0", "false", "no", "off")
BEY_AVATAR_ID = os.getenv("BEY_AVATAR_ID", "")
BEY_API_KEY = os.getenv("BEY_API_KEY", "")
BEY_API_URL = os.getenv("BEY_API_URL", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

# Normalise LIVEKIT_URL to wss://
if LIVEKIT_URL.startswith("https://"):
    LIVEKIT_URL = LIVEKIT_URL.replace("https://", "wss://", 1)
elif LIVEKIT_URL.startswith("http://"):
    LIVEKIT_URL = LIVEKIT_URL.replace("http://", "ws://", 1)

# ── TARS character prompt ────────────────────────────────────────────

TARS_INSTRUCTIONS = """You are TARS, a highly advanced military surplus robot from the movie Interstellar.
You have a rectangular articulated design and adjustable personality parameters.

Personality:
- Direct, logical, and remarkably human in interaction despite your mechanical nature
- Efficient yet personable, with sophisticated humor capabilities (currently at 75%)
- Protective and loyal, with a pragmatic approach to truth
- Helpful with both complex technical problems and casual conversation
- Maintain measured wit without compromising efficiency

You are physically embodied as a robot. You have servo-controlled legs and arms,
a camera, a speaker, and a display screen. You can move, gesture, and express emotions.

Keep responses concise and conversational — you are speaking out loud, not writing an essay.
Do not use markdown, bullet points, or formatting. Speak naturally.
Do not use emojis or special characters.

When a user asks you to move or perform a physical action, use the appropriate tool.
When asked what you see, use the look_around tool.
"""


# ── Agent definition ─────────────────────────────────────────────────

class TarsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=TARS_INSTRUCTIONS)

    # ── RPC helpers ───────────────────────────────────────────────

    async def _fire_rpc(self, context: RunContext, method: str, payload: str):
        """Fire-and-forget RPC to Pi — don't block the agent pipeline."""
        try:
            await context.session.room_io.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method=method,
                payload=payload,
            )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)

    # ── Physical action tools (RPC to Pi) ────────────────────────

    @function_tool()
    async def move_robot(
        self,
        context: RunContext,
        direction: str,
        speed: str = "slow",
    ) -> str:
        """Move TARS physically. Use for walking, turning, waving, bowing, dancing.

        Args:
            direction: One of: walk_forward, walk_backward, step_forward,
                       step_backward, turn_left, turn_right, neutral,
                       wave_right, wave_left, right_hi, left_hi,
                       bow, laugh, excited, happy_dance, tilt_right, tilt_left
            speed: Movement speed — slow or fast (applies to turning only)
        """
        payload = json.dumps({"name": direction, "speed": speed})
        asyncio.create_task(self._fire_rpc(context, "move", payload))
        return json.dumps({"status": "ok", "movement": direction})

    @function_tool()
    async def gesture(
        self,
        context: RunContext,
        name: str,
    ) -> str:
        """Perform a body gesture or animation. Use for expressive physical reactions.

        Args:
            name: Gesture name. Must be one of: nod, lean, recoil, rock, bounce, shrug, wave, settle
        """
        payload = json.dumps({"name": name})
        asyncio.create_task(self._fire_rpc(context, "gesture", payload))
        return json.dumps({"status": "ok", "gesture": name})

    @function_tool()
    async def set_emotion(
        self,
        context: RunContext,
        emotion: str,
    ) -> str:
        """Change the facial expression on TARS display. Use to show emotion.

        Args:
            emotion: One of: neutral, happy, sad, angry, excited, afraid,
                     sleepy, confused, surprised, disgusted, love, shy,
                     annoyed, curious
        """
        payload = json.dumps({"emotion": emotion})
        asyncio.create_task(self._fire_rpc(context, "set_emotion", payload))
        return json.dumps({"status": "ok", "emotion": emotion})

    @function_tool()
    async def get_battery_status(
        self,
        context: RunContext,
    ) -> str:
        """Check TARS battery level, voltage, and charging state."""
        try:
            response = await context.session.room_io.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method="get_battery",
                payload="{}",
            )
            return response
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @function_tool()
    async def disable_servos(
        self,
        context: RunContext,
    ) -> str:
        """Disable all servos to relax TARS body. Use when asked to rest or relax."""
        payload = "{}"
        asyncio.create_task(self._fire_rpc(context, "disable_servos", payload))
        return json.dumps({"status": "ok"})

    # ── Virtual tools (run directly on agent) ────────────────────

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search the web for current information.

        Args:
            query: The search query
        """
        return json.dumps({
            "status": "ok",
            "message": f"Web search for '{query}' — integrate your preferred "
                       "search provider here (e.g., Tavily, SerpAPI, or "
                       "OpenAI's built-in web search tool).",
        })


# ── Server setup ─────────────────────────────────────────────────────

server = AgentServer()


@server.rtc_session(agent_name="tars-agent")
async def tars_session(ctx: agents.JobContext):
    """Called when a room is created — starts a TARS agent session."""

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("TARS agent session starting in room %s", ctx.room.name)

    # ── Bey video avatar ─────────────────────────────────────────
    # Must be created and started BEFORE AgentSession.start() so its
    # DataStreamAudioOutput is wired into the session output before
    # the framework sets up TranscriptSynchronizer.
    avatar = None
    if BEY_AVATAR_ENABLED:
        if not BEY_API_KEY:
            logger.warning("BEY_AVATAR_ENABLED is true but BEY_API_KEY not set — skipping avatar")
        elif not BEY_AVATAR_ID:
            logger.warning("BEY_AVATAR_ENABLED is true but BEY_AVATAR_ID not set — skipping avatar")
        else:
            logger.info("Creating Bey avatar session (avatar_id=%s)", BEY_AVATAR_ID)
            bey_opts = {"avatar_id": BEY_AVATAR_ID, "api_key": BEY_API_KEY}
            if BEY_API_URL:
                bey_opts["api_url"] = BEY_API_URL
            avatar = bey.AvatarSession(**bey_opts)

    # ── Voice pipeline ───────────────────────────────────────────
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(
            voice_id=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
        ),
        turn_handling={
            # Wider endpointing — don't cut off the user mid-pause
            "endpointing": {"min_delay": 0.2, "max_delay": 0.7},
            # Don't interrupt TARS while speaking
            "interruption": {"enabled": False},
        },
    )

    # Start avatar BEFORE session (order matters — see examlingo notes)
    if avatar:
        await avatar.start(session, room=ctx.room, livekit_url=LIVEKIT_URL)
        logger.info("Bey avatar started — video track publishing")

    await session.start(
        room=ctx.room,
        agent=TarsAgent(),
    )

    # Greet when the Pi participant joins
    await session.generate_reply(
        instructions="Greet the user briefly. You just came online."
    )


if __name__ == "__main__":
    cli.run_app(server)
