# TARS Narration Mode (`app_narrate.py`)

A standalone voice-over tool for shooting TARS videos. You paste your script one
line at a time into the terminal; after a short countdown TARS speaks the line
aloud through the configured TTS voice, while the DSI screen shows the text. You
can drop inline `[gesture]` tags into a line to make the body move while it speaks.

It deliberately does **not** start speech-to-text, the LLM, wake-word detection,
the web ChatUI, or the idle body "fidgets". So TARS stays still and quiet between
takes and only does exactly what you type — ideal for recording.

---

## Running it

From the `src/` directory on the Pi:

```bash
python app_narrate.py
```

### Command-line options

All options are `key=value`, space-separated, in any order:

| Option          | Values                | Default    | Meaning                                            |
|-----------------|-----------------------|------------|----------------------------------------------------|
| `countdown`     | any whole number      | `3`        | Seconds counted down before each take (`0` = none) |
| `gestures`      | `on` / `off`          | `on`       | Whether `[gesture]` tags actually move the body    |
| `show_ui`       | `true` / `false`      | `true`     | Whether to show text on the DSI screen             |
| `port`          | a TCP port number     | `0` (off)  | Accept commands over the network instead of stdin  |
| `host`          | bind address          | `0.0.0.0`  | Interface the control server binds to               |
| `model`         | ElevenLabs model id   | `eleven_v3`| TTS model for narration (see Emotions below)       |

Examples:

```bash
python app_narrate.py countdown=5            # longer 5s countdown
python app_narrate.py gestures=off           # speak only, no body movement
python app_narrate.py show_ui=false          # audio only, screen off
python app_narrate.py countdown=0 gestures=off
python app_narrate.py port=5555              # control it remotely (see below)
python app_narrate.py model=inherit          # use config.ini's model instead of v3
```

---

## Controlling it remotely (recommended on the Pi)

When TARS runs on the Pi with the DSI display up, the launching terminal is tied
to the app — you can't also type script lines there without risking closing the
display. Instead, start it with a `port`, then send commands from **any other
terminal** (another SSH session, or your Mac on the same network).

**1. On the Pi**, launch with a control port. Use `tmux`/`nohup` so it keeps
running (and the display stays up) even if the launching SSH session closes:

```bash
cd ~/TARS-AI/src
tmux new -s narrate            # or: nohup python app_narrate.py port=5555 &
python app_narrate.py port=5555
# detach from tmux with: Ctrl-b then d   (leaves it running with the display up)
```

**2. From another terminal / your Mac**, connect and type lines live:

```bash
nc 192.168.178.70 5555
```

You'll get the same `script>` prompt and command set as the local REPL — type a
line, watch the countdown stream back, TARS speaks. Everything (`:replay`,
`:countdown N`, `:gest on|off`, `:help`, `:q`) works over the connection.

Notes:
- **One controller at a time.** The robot is a single device, so only one client
  is served at once; a second connection waits until the first disconnects.
- **Just leaving?** Close `nc` (Ctrl-C) to disconnect — the app keeps running and
  the display stays up, ready for the next connection.
- **`:q` stops the whole app** (and the display). Use it only when you're done.
- No password — only use this on a trusted home network. To restrict it to the
  local machine only, pass `host=127.0.0.1`.

The TTS voice/backend is whatever is set in `config.ini` under `[TTS] ttsoption`
(currently ElevenLabs). This tool does not change it.

---

## Using it (the REPL)

Once running you get a `script>` prompt. Type (or paste) a line and press Enter.
It counts down in the terminal, then speaks.

### Commands

| Command          | What it does                                             |
|------------------|----------------------------------------------------------|
| `<your text>`    | Speak a line (after the countdown).                      |
| `:replay`        | Re-speak the previous line (same text + gestures).       |
| `:countdown N`   | Change the countdown length on the fly (`:countdown 5`). |
| `:gest on`       | Turn gesture playback on.                                |
| `:gest off`      | Turn gesture playback off.                               |
| `:ui`            | Print the current settings.                              |
| `:help`          | Show the command list + available gestures.              |
| `:q` / `:quit`   | Exit narration mode. (Ctrl-C / Ctrl-D also exit.)        |

---

## Gesture tags

Put a gesture in square brackets anywhere in a line. The tag is removed from what
TARS *says* and instead triggers the movement as the line begins:

```
script> [wave] Hello there. [nod] I am TARS.
```

That speaks "Hello there. I am TARS." and plays the `wave` then `nod` gestures.

Only one gesture physically runs at a time (they share a single body lock), so if
you stack several on one line they queue up. For a single, clean movement per take,
one tag per line works best.

### Available gestures

- `nod`     — agreement, acknowledgment
- `lean`    — curiosity, leaning in with interest
- `recoil`  — surprise, shock, disbelief
- `rock`    — disagreement, thinking, "no way"
- `bounce`  — joy, excitement, laughter
- `shrug`   — uncertainty, "who knows"
- `wave`    — greeting, hello / goodbye
- `settle`  — calm acceptance, respect, gratitude

(If you type a gesture name that isn't in this list, it's treated as plain text
and left in the spoken line — see below.)

---

## Emotions (ElevenLabs v3 audio tags)

Narration runs on the **`eleven_v3`** model by default, which understands inline
**audio tags** — bracketed emotion/delivery cues — right in the text:

```
script> [wave] Hey. [curious] So... [whispers] here's the part they don't tell you.
```

Common audio tags:

- Emotion: `[excited]`, `[nervous]`, `[frustrated]`, `[sad]`, `[calm]`, `[angry]`
- Reactions: `[laughs]`, `[sighs]`, `[gasps]`, `[gulps]`, `[whispers]`
- Delivery: `[sarcastic]`, `[cheerfully]`, `[deadpan]`, `[playfully]`, `[hesitates]`

These pass straight through to ElevenLabs because the tool only pulls out tags
that match a **gesture** name (see list above) — every other `[tag]` is left in
the spoken text. None of the emotion tags collide with gesture names, so you can
mix them freely: `[wave]` moves the body, `[excited]` colors the voice.

**How it's wired:** this is a standalone process, so it sets `eleven_v3` just for
narration — your main TARS app keeps whatever model is in `config.ini`. Override
per run with `model=<id>`, or `model=inherit` to use the config model.

### Pauses on v3

v3 does **not** use SSML `<break time="1s"/>` tags — they're ignored. For pauses,
use ellipses or an audio tag instead:

```
script> Give me a moment... [pauses] okay, I'm ready.
```

### Notes / caveats

- v3 is tuned for expressiveness, not low latency — fine here since we're not live.
- v3 is most consistent with a little more text per line; very short lines can
  vary take-to-take, so `:replay` until you like it.
- Make sure your configured voice is v3-compatible. If v3 misbehaves, fall back
  with `model=eleven_multilingual_v2` (no audio tags, but stable).

---

## Typical recording workflow

1. `python app_narrate.py` (optionally with `countdown=5`).
2. Get your camera rolling.
3. Paste the first line at `script>` and hit Enter.
4. Watch the terminal countdown; TARS speaks on "SPEAKING".
5. If the take is bad, type `:replay` and go again.
6. Move to the next line. Repeat.
7. `:q` when finished.
