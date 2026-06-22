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

    # ── Physical action tools (RPC to Pi) ────────────────────────

    @function_tool()
    async def move_robot(
        self,
        context: RunContext,
        direction: str,
        speed: str = "slow",
    ) -> str:
        """Move TARS in a direction. Use this when asked to walk, step, or turn.

        Args:
            direction: One of: walk_forward, walk_backward, step_forward,
                       step_backward, turn_left, turn_right, neutral
            speed: Movement speed — slow or fast
        """
        payload = json.dumps({"name": direction, "speed": speed})
        try:
            response = await context.session.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method="move",
                payload=payload,
            )
            return response
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @function_tool()
    async def gesture(
        self,
        context: RunContext,
        name: str,
    ) -> str:
        """Perform a body gesture or animation. Use for expressive physical reactions.

        Args:
            name: Gesture name — nod, lean, recoil, rock, bounce, shrug, wave,
                  settle, bow, laugh, excited, happy_dance, wave_right, wave_left,
                  right_hi, left_hi, tilt_right, tilt_left
        """
        payload = json.dumps({"name": name})
        try:
            response = await context.session.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method="gesture",
                payload=payload,
            )
            return response
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

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
        try:
            response = await context.session.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method="set_emotion",
                payload=payload,
            )
            return response
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @function_tool()
    async def get_battery_status(
        self,
        context: RunContext,
    ) -> str:
        """Check TARS battery level, voltage, and charging state."""
        try:
            response = await context.session.room.local_participant.perform_rpc(
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
        try:
            response = await context.session.room.local_participant.perform_rpc(
                destination_identity="tars-pi",
                method="disable_servos",
                payload="{}",
            )
            return response
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

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
        if not os.getenv("BEY_API_KEY"):
            logger.warning("BEY_AVATAR_ENABLED is true but BEY_API_KEY not set — skipping avatar")
        elif not BEY_AVATAR_ID:
            logger.warning("BEY_AVATAR_ENABLED is true but BEY_AVATAR_ID not set — skipping avatar")
        else:
            logger.info("Creating Bey avatar session (avatar_id=%s)", BEY_AVATAR_ID)
            avatar = bey.AvatarSession(avatar_id=BEY_AVATAR_ID)

    # ── Voice pipeline ───────────────────────────────────────────
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(
            voice_id=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
        ),
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
