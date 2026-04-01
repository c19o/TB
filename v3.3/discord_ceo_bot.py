"""
Discord CEO Bot — Savage22 Server Command Center
=================================================
Persistent Discord bot that gives you full control over the autonomous
training company. Bidirectional: reports to you AND takes your orders.

INFRASTRUCTURE ONLY — does NOT import or modify any trading/ML code.

Commands (via Discord DM):
    !status          — Full status of all training + agents
    !agents          — List active Paperclip agents and their tasks
    !logs [n]        — Last n events from discord_gate_log.json
    !approve [id]    — Approve a pending gate
    !deny [id]       — Deny a pending gate
    !pause           — Pause all autonomous operations
    !resume          — Resume autonomous operations
    !budget          — Show cloud spend so far
    !machine         — Show active vast.ai machines
    !progress [tf]   — Training progress for a timeframe
    !artifact [tf]   — List downloaded artifacts
    !loopme [event]  — Put yourself in the loop for an event type
    !unloopme [event]— Remove yourself from the loop for an event
    !concurrent [N]  — Get/set max concurrent agents (1-20, default 5)
    !priority [task] — Set a task as highest priority
    !stop [agent]    — Stop a specific agent
    !help            — Show all commands

Environment variables:
    DISCORD_BOT_TOKEN   — Bot token
    DISCORD_USER_ID     — Your Discord user ID (for DM auth)

Usage:
    python discord_ceo_bot.py
"""

import os
import sys
import json
import time
import signal
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

try:
    import requests
except ImportError:
    print("ERROR: 'requests' required. pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "https://discord.com/api/v10"
GATEWAY_URL = None  # Fetched dynamically
POLL_INTERVAL = 2.0  # seconds between message checks
HEARTBEAT_INTERVAL = 60  # seconds between heartbeat status checks
BOT_PREFIX = "!"
CEO_WORKSPACE = Path.home() / "claude-ceo" / "workspace"

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "discord_gate_log.json"
STATE_FILE = SCRIPT_DIR / "ceo_bot_state.json"
SESSION_RESUME = SCRIPT_DIR / "SESSION_RESUME.md"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(SCRIPT_DIR / "discord_ceo_bot.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("ceo_bot")

# ---------------------------------------------------------------------------
# Bot State
# ---------------------------------------------------------------------------

class BotState:
    """Persistent state for the CEO bot."""

    def __init__(self):
        self.paused = False
        self.loop_events: set = set()  # Events user wants to be in the loop for
        self.pending_approvals: dict = {}  # id -> {event, context, timestamp}
        self.cloud_spend_usd: float = 0.0
        self.max_concurrent_agents: int = 5  # Max agents running simultaneously (1-20)
        self.active_machines: list = []
        self.training_status: dict = {
            "1w": {"status": "COMPLETE", "accuracy": "57.5%", "binary": "79.3%"},
            "1d": {"status": "BLOCKED", "blocker": "daemon RELOAD bug"},
            "4h": {"status": "QUEUED", "blocker": "waiting on 1d"},
            "1h": {"status": "QUEUED", "blocker": "waiting on 4h"},
            "15m": {"status": "QUEUED", "blocker": "user picks machine"},
        }
        self.agents: dict = {}  # agent_id -> {role, status, task, last_heartbeat}
        self._load()

    def _load(self):
        """Load state from disk."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                self.paused = data.get("paused", False)
                self.loop_events = set(data.get("loop_events", []))
                self.cloud_spend_usd = data.get("cloud_spend_usd", 0.0)
                self.max_concurrent_agents = data.get("max_concurrent_agents", 5)
                self.active_machines = data.get("active_machines", [])
                self.training_status = data.get("training_status", self.training_status)
                self.agents = data.get("agents", {})
                logger.info("Loaded bot state from disk")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load state: {e}")

    def save(self):
        """Persist state to disk."""
        data = {
            "paused": self.paused,
            "loop_events": list(self.loop_events),
            "cloud_spend_usd": self.cloud_spend_usd,
            "max_concurrent_agents": self.max_concurrent_agents,
            "active_machines": self.active_machines,
            "training_status": self.training_status,
            "agents": self.agents,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        STATE_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Discord REST Client
# ---------------------------------------------------------------------------

class DiscordClient:
    """Minimal Discord REST client."""

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Savage22-CEO-Bot/1.0",
        })

    def _request(self, method: str, path: str, **kwargs):
        url = f"{API_BASE}{path}"
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5)
            logger.warning(f"Rate limited, sleeping {retry_after}s")
            time.sleep(retry_after)
            resp = self.session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            logger.error(f"Discord API error {resp.status_code}: {body}")
            return None
        if resp.status_code == 204:
            return {}
        return resp.json()

    def get_dm_channel(self, user_id: str) -> Optional[str]:
        data = self._request("POST", "/users/@me/channels", json={"recipient_id": user_id})
        return data["id"] if data else None

    def send_message(self, channel_id: str, content: str = None, embed: dict = None) -> Optional[dict]:
        payload = {}
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]
        return self._request("POST", f"/channels/{channel_id}/messages", json=payload)

    def get_messages(self, channel_id: str, after: str = None, limit: int = 10) -> list:
        params = {"limit": limit}
        if after:
            params["after"] = after
        result = self._request("GET", f"/channels/{channel_id}/messages", params=params)
        return result if isinstance(result, list) else []

    def get_bot_user(self) -> Optional[dict]:
        return self._request("GET", "/users/@me")


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

def _read_ceo_live():
    """Read live data from CEO workspace."""
    queue_dir = CEO_WORKSPACE / "queue"
    session_dir = CEO_WORKSPACE / "sessions"
    tasks = []
    sessions = {}

    if queue_dir.exists():
        for f in queue_dir.glob("*.json"):
            try:
                tasks.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    if session_dir.exists():
        for f in session_dir.glob("*.json"):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
                sessions[f.stem] = s
            except Exception:
                pass
    return tasks, sessions


def cmd_status(state: BotState, args: str) -> dict:
    """Full live status from CEO system + training state."""
    tasks, sessions = _read_ceo_live()

    running = [t for t in tasks if t.get("status") == "running"]
    completed = [t for t in tasks if t.get("status") == "completed"]
    pending = [t for t in tasks if t.get("status") == "pending"]
    total_cost = sum(s.get("cost_usd", 0) or 0 for s in sessions.values())

    lines = ["**SAVAGE22 CEO DASHBOARD**\n"]

    # Training status
    lines.append("**Training Status:**")
    for tf, info in state.training_status.items():
        emoji = {"COMPLETE": "✅", "BLOCKED": "🔴", "QUEUED": "⏸️", "RUNNING": "🔄"}.get(info["status"], "❓")
        line = f"{emoji} **{tf}**: {info['status']}"
        if info.get("accuracy"):
            line += f" | {info['accuracy']}"
        if info.get("blocker"):
            line += f" | {info['blocker']}"
        lines.append(line)

    # Live agent activity
    lines.append(f"\n**Agents:** {len(running)} running | {len(completed)} done | {len(pending)} queued")

    if running:
        lines.append("\n**🔄 RUNNING NOW:**")
        for t in running:
            name = t.get("task_name", "?")
            role = t.get("role", "?")
            sid = t.get("session_id", "")
            elapsed = ""
            if sid and sid in sessions:
                s = sessions[sid]
                if s.get("started_at"):
                    try:
                        from datetime import datetime, timezone
                        started = datetime.fromisoformat(s["started_at"].replace("Z", "+00:00"))
                        elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
                        elapsed = f" ({int(elapsed_s)}s)"
                    except Exception:
                        pass
            lines.append(f"  ⚡ **{name}** [{role}]{elapsed}")

    if completed:
        recent = sorted(completed, key=lambda t: t.get("completed_at", ""), reverse=True)[:5]
        lines.append("\n**✅ RECENTLY COMPLETED:**")
        for t in recent:
            name = t.get("task_name", "?")
            sid = t.get("session_id", "")
            cost = ""
            if sid and sid in sessions:
                c = sessions[sid].get("cost_usd")
                if c:
                    cost = f" ${c:.2f}"
            lines.append(f"  ✅ {name}{cost}")

    lines.append(f"\n**Total CEO Cost**: ${total_cost:.2f}")
    lines.append(f"**Max Concurrent**: {state.max_concurrent_agents}")
    lines.append(f"**Mode**: {'PAUSED ⏸️' if state.paused else 'RUNNING ▶️'}")

    return {"content": "\n".join(lines)}


def cmd_agents(state: BotState, args: str) -> dict:
    """List active agents from live CEO data."""
    tasks, sessions = _read_ceo_live()

    running = [t for t in tasks if t.get("status") == "running"]
    if not running:
        return {"content": "No agents currently running. Queue may be empty or paused."}

    lines = ["**ACTIVE AGENTS**\n"]
    for t in running:
        name = t.get("task_name", "?")
        role = t.get("role", "?")
        sid = t.get("session_id", "")
        elapsed = ""
        pid = ""
        if sid and sid in sessions:
            s = sessions[sid]
            if s.get("pid"):
                pid = f" PID:{s['pid']}"
            if s.get("started_at"):
                try:
                    from datetime import datetime, timezone
                    started = datetime.fromisoformat(s["started_at"].replace("Z", "+00:00"))
                    elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
                    if elapsed_s > 3600:
                        elapsed = f" ({int(elapsed_s/3600)}h{int(elapsed_s%3600/60)}m)"
                    elif elapsed_s > 60:
                        elapsed = f" ({int(elapsed_s/60)}m{int(elapsed_s%60)}s)"
                    else:
                        elapsed = f" ({int(elapsed_s)}s)"
                except Exception:
                    pass
        lines.append(f"⚡ **{role}** — {name}{elapsed}{pid}")

    lines.append(f"\n**Total running**: {len(running)}")
    return {"content": "\n".join(lines)}


def cmd_logs(state: BotState, args: str) -> dict:
    """Show recent gate log entries."""
    n = 10
    if args.strip().isdigit():
        n = int(args.strip())

    if not LOG_FILE.exists():
        return {"content": "No gate log entries yet."}

    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"content": "Failed to read gate log."}

    entries = data[-n:]
    lines = [f"**Last {len(entries)} Gate Events**\n"]
    for entry in entries:
        ts = entry.get("logged_at", "?")[:19]
        level = entry.get("level", "?")
        event = entry.get("event_type", "?")
        result = entry.get("result", entry.get("sent", "?"))
        msg = entry.get("message", "")[:80]
        lines.append(f"`{ts}` [{level}] **{event}**: {msg} → {result}")

    return {"content": "\n".join(lines)}


def cmd_pause(state: BotState, args: str) -> dict:
    """Pause all autonomous operations."""
    state.paused = True
    state.save()
    return {"content": "⏸️ **All autonomous operations PAUSED.** Agents will complete current tasks but won't start new ones.\nUse `!resume` to continue."}


def cmd_resume(state: BotState, args: str) -> dict:
    """Resume autonomous operations."""
    state.paused = False
    state.save()
    return {"content": "▶️ **Autonomous operations RESUMED.** Agents will pick up where they left off."}


def cmd_budget(state: BotState, args: str) -> dict:
    """Show cloud spend."""
    lines = [f"**Cloud Budget**\n"]
    lines.append(f"Total spent: **${state.cloud_spend_usd:.2f}**")
    for m in state.active_machines:
        name = m.get("name", "unknown")
        cost_hr = m.get("cost_hr", "?")
        hours = m.get("hours_running", "?")
        lines.append(f"  {name}: {cost_hr}/hr × {hours}h")
    return {"content": "\n".join(lines)}


def cmd_machine(state: BotState, args: str) -> dict:
    """Show active machines."""
    if not state.active_machines:
        return {"content": "No active machines. Sichuan (33876301) is PAUSED."}

    lines = ["**Active Machines**\n"]
    for m in state.active_machines:
        lines.append(f"• **{m.get('name', '?')}** (ID: {m.get('id', '?')}) — {m.get('status', '?')} — {m.get('cost_hr', '?')}/hr")
    return {"content": "\n".join(lines)}


def cmd_progress(state: BotState, args: str) -> dict:
    """Training progress for a specific timeframe."""
    tf = args.strip().lower()
    if not tf:
        return {"content": "Usage: `!progress <tf>` — e.g. `!progress 1d`"}

    info = state.training_status.get(tf)
    if not info:
        return {"content": f"Unknown timeframe: {tf}. Valid: 1w, 1d, 4h, 1h, 15m"}

    lines = [f"**{tf.upper()} Training Progress**\n"]
    for k, v in info.items():
        lines.append(f"**{k}**: {v}")
    return {"content": "\n".join(lines)}


def cmd_loopme(state: BotState, args: str) -> dict:
    """Put user in the loop for an event type."""
    event = args.strip().lower()
    if not event:
        return {"content": "Usage: `!loopme <event>` — e.g. `!loopme bug_found`\nAvailable events: rent_machine, destroy_machine, bug_found, code_change_core, matrix_thesis_change, training_complete, oom_error, pipeline_error"}

    state.loop_events.add(event)
    state.save()
    return {"content": f"🔔 You're now **in the loop** for `{event}`. I'll ask for your approval before proceeding."}


def cmd_unloopme(state: BotState, args: str) -> dict:
    """Remove user from the loop for an event type."""
    event = args.strip().lower()
    if not event:
        return {"content": f"Usage: `!unloopme <event>`\nCurrently in loop for: {', '.join(state.loop_events) or 'nothing'}"}

    state.loop_events.discard(event)
    state.save()
    return {"content": f"🔕 You're now **out of the loop** for `{event}`. Agents will handle it autonomously."}


def cmd_help(state: BotState, args: str) -> dict:
    """Show all commands."""
    return {"content": """**SAVAGE22 CEO BOT — COMMANDS**

**Status & Monitoring**
`!status` — Full training status overview
`!agents` — List active agents and their tasks
`!logs [n]` — Last n gate events (default 10)
`!progress <tf>` — Training progress for timeframe
`!budget` — Cloud spend summary
`!machine` — Active vast.ai machines
`!artifact <tf>` — List downloaded artifacts

**Control**
`!pause` — Pause all autonomous operations
`!resume` — Resume autonomous operations
`!killswitch [on|off|status]` — Emergency halt for live trader (SAV-19)
`!approve <id>` — Approve a pending gate
`!deny <id>` — Deny a pending gate
`!concurrent [N]` — Get/set max concurrent agents (1-20)
`!priority <task>` — Set highest priority task
`!stop <agent>` — Stop a specific agent

**Loop Control**
`!loopme <event>` — Get approval requests for event type
`!unloopme <event>` — Let agents handle event autonomously

**Events**: rent_machine, destroy_machine, bug_found, code_change_core, matrix_thesis_change, training_complete, oom_error, pipeline_error

Or just type naturally — I'll understand questions like "how's 1d going?" or "what's the blocker?"
"""}


def cmd_artifact(state: BotState, args: str) -> dict:
    """List artifacts for a timeframe."""
    tf = args.strip().lower()
    if not tf:
        return {"content": "Usage: `!artifact <tf>` — e.g. `!artifact 1w`"}

    artifact_dir = SCRIPT_DIR / f"{tf}_cloud_artifacts"
    artifact_dir_v2 = SCRIPT_DIR / f"{tf}_cloud_artifacts_v2"
    artifact_dir_v3 = SCRIPT_DIR / f"{tf}_cloud_artifacts_v3"

    found = []
    for d in [artifact_dir, artifact_dir_v2, artifact_dir_v3]:
        if d.exists():
            files = sorted(d.iterdir())
            for f in files:
                size = f.stat().st_size
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f} MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.1f} KB"
                else:
                    size_str = f"{size} B"
                found.append(f"`{d.name}/{f.name}` ({size_str})")

    if not found:
        return {"content": f"No artifacts found for {tf}."}

    return {"content": f"**{tf.upper()} Artifacts**\n" + "\n".join(found)}


def cmd_concurrent(state: BotState, args: str) -> dict:
    """Get or set max concurrent agent count."""
    arg = args.strip()
    if not arg:
        return {"content": f"\U0001f527 Current max concurrent: {state.max_concurrent_agents}"}

    if not arg.isdigit():
        return {"content": f"\u274c Invalid value: `{arg}`. Must be a number 1-20."}

    n = int(arg)
    if n < 1 or n > 20:
        return {"content": f"\u274c Value {n} out of range. Must be 1-20."}

    state.max_concurrent_agents = n
    state.save()
    return {"content": f"\u2705 Max concurrent agents set to {n}"}


def cmd_killswitch(state: BotState, args: str) -> dict:
    """Emergency kill switch for live trader. Creates/removes KILL_SWITCH file."""
    killswitch_path = SCRIPT_DIR / "KILL_SWITCH"
    args_lower = args.strip().lower()

    # Status check
    if args_lower in ("status", "check", ""):
        is_active = killswitch_path.exists()
        if is_active:
            mtime = killswitch_path.stat().st_mtime
            from datetime import datetime
            activated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            return {"content": f"🔴 **KILL SWITCH ACTIVE**\nActivated: {activated_at}\nLive trading is **HALTED**.\nUse `!killswitch off` to resume."}
        else:
            return {"content": "✅ **KILL SWITCH INACTIVE**\nLive trading is **RUNNING**.\nUse `!killswitch on` to halt all trading immediately."}

    # Enable kill switch
    if args_lower in ("on", "enable", "activate", "halt", "stop"):
        if killswitch_path.exists():
            return {"content": "⚠️ Kill switch is already active."}
        try:
            killswitch_path.write_text(f"HALT - Activated via Discord at {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
            return {"content": "🔴 **KILL SWITCH ACTIVATED**\n\nAll live trading is now **HALTED**.\nThe trader will:\n- Stop opening new positions immediately\n- Keep existing positions open (you can close manually)\n- Check every 10 seconds and wait\n\nUse `!killswitch off` to resume trading."}
        except Exception as e:
            return {"content": f"❌ Failed to activate kill switch: {e}"}

    # Disable kill switch
    if args_lower in ("off", "disable", "deactivate", "resume", "go"):
        if not killswitch_path.exists():
            return {"content": "⚠️ Kill switch is not active."}
        try:
            killswitch_path.unlink()
            return {"content": "✅ **KILL SWITCH DEACTIVATED**\n\nLive trading has **RESUMED**.\nThe trader will pick up on the next 15-minute cycle."}
        except Exception as e:
            return {"content": f"❌ Failed to deactivate kill switch: {e}"}

    # Unknown argument
    return {"content": f"❓ Unknown argument: `{args}`\n\nUsage:\n`!killswitch` or `!killswitch status` — Check status\n`!killswitch on` — Halt all trading\n`!killswitch off` — Resume trading"}


# Command registry
COMMANDS: dict[str, Callable] = {
    "status": cmd_status,
    "agents": cmd_agents,
    "logs": cmd_logs,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "budget": cmd_budget,
    "machine": cmd_machine,
    "progress": cmd_progress,
    "loopme": cmd_loopme,
    "unloopme": cmd_unloopme,
    "help": cmd_help,
    "artifact": cmd_artifact,
    "concurrent": cmd_concurrent,
    "killswitch": cmd_killswitch,
}


# ---------------------------------------------------------------------------
# Natural Language Handler
# ---------------------------------------------------------------------------

def handle_natural_language(state: BotState, message: str) -> Optional[dict]:
    """Handle natural language queries that aren't commands."""
    msg = message.lower().strip()

    # Status queries
    if any(w in msg for w in ["status", "how's it going", "what's happening", "update", "whats up", "how are things"]):
        return cmd_status(state, "")

    # Specific TF queries
    for tf in ["1w", "1d", "4h", "1h", "15m"]:
        if tf in msg and any(w in msg for w in ["how", "progress", "status", "going", "blocker", "stuck"]):
            return cmd_progress(state, tf)

    # Agent queries
    if any(w in msg for w in ["agents", "who's working", "team"]):
        return cmd_agents(state, "")

    # Budget queries
    if any(w in msg for w in ["budget", "cost", "spend", "money", "how much"]):
        return cmd_budget(state, "")

    # Machine queries
    if any(w in msg for w in ["machine", "gpu", "vast", "server", "cloud"]):
        return cmd_machine(state, "")

    # Pause/resume
    if any(w in msg for w in ["pause", "stop everything", "hold on", "wait"]):
        return cmd_pause(state, "")
    if any(w in msg for w in ["resume", "continue", "go ahead", "start again", "unpause"]):
        return cmd_resume(state, "")

    return None


# ---------------------------------------------------------------------------
# Main Bot Loop (REST-based polling, no WebSocket needed)
# ---------------------------------------------------------------------------

class CEOBot:
    """
    Persistent Discord bot using REST polling.
    Simpler than WebSocket gateway, works behind firewalls, no special ports.
    """

    def __init__(self):
        self.token = os.environ.get("DISCORD_BOT_TOKEN")
        self.user_id = os.environ.get("DISCORD_USER_ID")

        if not self.token:
            logger.error("DISCORD_BOT_TOKEN not set")
            sys.exit(1)
        if not self.user_id:
            logger.error("DISCORD_USER_ID not set")
            sys.exit(1)

        self.client = DiscordClient(self.token)
        self.state = BotState()
        self.dm_channel_id: Optional[str] = None
        self.last_message_id: Optional[str] = None
        self.bot_user_id: Optional[str] = None
        self._running = True

    def start(self):
        """Initialize and start the bot."""
        logger.info("Starting CEO Bot...")

        # Get bot user info
        bot_user = self.client.get_bot_user()
        if not bot_user:
            logger.error("Failed to get bot user info")
            sys.exit(1)
        self.bot_user_id = bot_user["id"]
        logger.info(f"Bot user: {bot_user['username']}#{bot_user.get('discriminator', '0')}")

        # Open DM channel
        self.dm_channel_id = self.client.get_dm_channel(self.user_id)
        if not self.dm_channel_id:
            logger.error("Failed to open DM channel")
            sys.exit(1)
        logger.info(f"DM channel: {self.dm_channel_id}")

        # Get last message ID to avoid replaying history
        messages = self.client.get_messages(self.dm_channel_id, limit=1)
        if messages:
            self.last_message_id = messages[0]["id"]
        logger.info(f"Last message ID: {self.last_message_id}")

        # Send startup message
        self.client.send_message(self.dm_channel_id, content="🟢 **CEO Bot Online.** Type `!help` for commands or just ask me anything.")

        # Main loop
        logger.info("Entering main loop...")
        try:
            while self._running:
                self._poll_messages()
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.client.send_message(self.dm_channel_id, content="🔴 **CEO Bot shutting down.** Operations continue in background.")

    def _poll_messages(self):
        """Check for new DM messages from the user."""
        messages = self.client.get_messages(
            self.dm_channel_id,
            after=self.last_message_id,
            limit=10,
        )

        if not messages:
            return

        # Messages come newest-first, reverse for chronological processing
        messages.sort(key=lambda m: m["id"])

        for msg in messages:
            self.last_message_id = msg["id"]

            # Ignore bot's own messages
            if msg["author"]["id"] == self.bot_user_id:
                continue

            # Only process messages from the authorized user
            if str(msg["author"]["id"]) != self.user_id:
                continue

            content = msg.get("content", "").strip()
            if not content:
                continue

            logger.info(f"User message: {content}")
            try:
                self._handle_message(content)
            except Exception as e:
                logger.error(f"Handle message error: {e}", exc_info=True)
                try:
                    self.client.send_message(self.dm_channel_id, content=f"❌ Internal error: {e}")
                except Exception:
                    pass

    def _handle_message(self, content: str):
        """Route a user message to the appropriate handler."""
        # Check for command prefix
        if content.startswith(BOT_PREFIX):
            cmd_text = content[len(BOT_PREFIX):].strip()
            parts = cmd_text.split(None, 1)
            cmd_name = parts[0].lower() if parts else ""
            cmd_args = parts[1] if len(parts) > 1 else ""

            handler = COMMANDS.get(cmd_name)
            if handler:
                try:
                    response = handler(self.state, cmd_args)
                    if response:
                        self.client.send_message(self.dm_channel_id, **response)
                except Exception as e:
                    logger.error(f"Command error: {e}", exc_info=True)
                    self.client.send_message(self.dm_channel_id, content=f"❌ Error running `{cmd_name}`: {e}")
            else:
                self.client.send_message(self.dm_channel_id, content=f"Unknown command: `{cmd_name}`. Type `!help` for available commands.")
            return

        # Try natural language
        response = handle_natural_language(self.state, content)
        if response:
            self.client.send_message(self.dm_channel_id, **response)
            return

        # Default response
        self.client.send_message(
            self.dm_channel_id,
            content=f"I heard you, but I'm not sure what you're asking. Type `!help` for commands.\n\nYou said: *{content[:200]}*"
        )

    def stop(self):
        """Stop the bot gracefully."""
        self._running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = CEOBot()

    # Handle graceful shutdown
    def _signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        bot.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    bot.start()
