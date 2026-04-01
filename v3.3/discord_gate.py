"""
Discord Gate Manager — Savage22 Server Infrastructure
=====================================================
Standalone notification/approval system using Discord REST API v10.
Does NOT import or modify any trading/ML code.

Usage:
    from discord_gate import gate

    gate.notify("Training 1d complete", {"accuracy": "62.3%", "pbo": 0.12})
    approved = gate.approve("rent_machine", {...})
    gate.critical("OOM at 700GB", {"step": "cross_gen", "tf": "1h"})

Environment variables:
    DISCORD_BOT_TOKEN   - Bot token for Discord API
    DISCORD_USER_ID     - Your Discord user ID (for DMs)
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://discord.com/api/v10"

# Notification levels
INFO = "INFO"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
CRITICAL = "CRITICAL"

# Color codes for embeds (decimal, not hex)
COLOR_GREEN = 0x2ECC71    # INFO
COLOR_YELLOW = 0xF1C40F   # APPROVAL_REQUIRED
COLOR_RED = 0xE74C3C      # CRITICAL

# Timing
APPROVAL_TIMEOUT_S = 30 * 60        # 30 minutes before reminder
APPROVAL_ESCALATE_S = 2 * 60 * 60   # 2 hours before escalation to CRITICAL
CRITICAL_PING_INTERVAL_S = 5 * 60   # 5 minutes between critical pings
POLL_INTERVAL_S = 5                  # How often to check for reactions/replies

# Priority tiers
P0_HALT = "P0"          # System halt: kill switch, OOM
P1_INVESTIGATE = "P1"   # Requires investigation: feature stale, model drift
P2_SUMMARY = "P2"       # Summary/FYI: cost anomaly, optimization proposals

# Priority to level mapping
PRIORITY_TO_LEVEL = {
    P0_HALT: CRITICAL,
    P1_INVESTIGATE: APPROVAL_REQUIRED,
    P2_SUMMARY: INFO,
}

# Rate limiting
RATE_LIMIT_MAX = 3       # Max alerts per window
RATE_LIMIT_WINDOW_S = 5 * 60  # 5 minutes

# Approval reactions
REACTION_APPROVE = "\u2705"  # white_check_mark
REACTION_DENY = "\u274c"     # x

# Text responses mapped to approval/denial
APPROVE_WORDS = {"yes", "y", "go", "go for it", "approved", "approve", "ok", "do it", "proceed", "confirmed", "confirm", "ship it", "send it"}
DENY_WORDS = {"no", "n", "stop", "deny", "denied", "cancel", "abort", "nope", "dont", "don't", "hold", "wait"}

# Hard-gated events (APPROVAL_REQUIRED by default)
HARD_GATED_EVENTS = {
    "rent_machine", "machine_rental",
    "code_change_core", "core_file_change",
    "matrix_thesis_change",
    "destroy_machine", "machine_destroy",
}

# Auto-approved events (INFO by default)
AUTO_APPROVED_EVENTS = {
    "training_complete", "training_progress", "training_step",
    "artifact_download", "download_complete",
    "session_resume", "session_update",
}

# Core files that trigger hard gate on changes
CORE_FILES = {
    "feature_library.py",
    "config.py",
    "validate.py",
    "v2_cross_generator.py",
}

# Event type display names, descriptions, and default priorities
EVENT_DISPLAY = {
    "rent_machine":         ("Machine Rental Request", "Approval needed to rent cloud GPU", P1_INVESTIGATE),
    "machine_rental":       ("Machine Rental Request", "Approval needed to rent cloud GPU", P1_INVESTIGATE),
    "code_change_core":     ("Core Code Change", "A core file is being modified", P1_INVESTIGATE),
    "core_file_change":     ("Core Code Change", "A core file is being modified", P1_INVESTIGATE),
    "matrix_thesis_change": ("Matrix Thesis Impact", "Change may affect the matrix thesis", P1_INVESTIGATE),
    "destroy_machine":      ("Machine Destruction", "Approval needed to destroy cloud machine", P1_INVESTIGATE),
    "machine_destroy":      ("Machine Destruction", "Approval needed to destroy cloud machine", P1_INVESTIGATE),
    "training_complete":    ("Training Complete", "A training step has finished", P2_SUMMARY),
    "training_progress":    ("Training Progress", "Training update", P2_SUMMARY),
    "training_step":        ("Training Step", "Training step completed", P2_SUMMARY),
    "artifact_download":    ("Artifact Downloaded", "Artifact saved successfully", P2_SUMMARY),
    "download_complete":    ("Download Complete", "Download finished", P2_SUMMARY),
    "session_resume":       ("Session Resume", "Session resume info updated", P2_SUMMARY),
    "session_update":       ("Session Update", "Session state changed", P2_SUMMARY),
    "bug_found":            ("Bug Found", "A bug was detected", P1_INVESTIGATE),
    "milestone_complete":   ("Milestone Complete", "A milestone was reached", P2_SUMMARY),
    "oom_error":            ("Out of Memory", "OOM error occurred", P0_HALT),
    "pipeline_error":       ("Pipeline Error", "Pipeline encountered an error", P1_INVESTIGATE),
    # New event types for tiering
    "kill_switch":          ("Kill Switch Activated", "Emergency system halt triggered", P0_HALT),
    "feature_stale":        ("Feature Data Stale", "Feature database is outdated", P1_INVESTIGATE),
    "model_drift":          ("Model Drift Detected", "Model performance degraded", P1_INVESTIGATE),
    "cost_anomaly":         ("Cost Anomaly", "Unexpected cost spike detected", P2_SUMMARY),
    "self_improvement":     ("Self-Improvement Proposal", "Agent suggests optimization", P2_SUMMARY),
}

# Log file path (same directory as this script)
LOG_FILE = Path(__file__).parent / "discord_gate_log.json"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("discord_gate")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] discord_gate %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Persistent log
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def _append_log(entry: dict) -> None:
    """Append an entry to discord_gate_log.json (thread-safe)."""
    entry["logged_at"] = datetime.now(timezone.utc).isoformat()
    with _log_lock:
        try:
            if LOG_FILE.exists():
                data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            else:
                data = []
        except (json.JSONDecodeError, OSError):
            data = []
        data.append(entry)
        LOG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Discord REST helpers
# ---------------------------------------------------------------------------

class DiscordAPIError(Exception):
    """Raised when a Discord API call fails."""
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Discord API {status_code}: {body}")


class DiscordClient:
    """Thin wrapper around Discord REST API v10 using requests."""

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Savage22-GateManager/1.0",
        })

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{API_BASE}{path}"
        resp = self.session.request(method, url, **kwargs)
        # Handle rate limits
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5)
            logger.warning(f"Rate limited, waiting {retry_after}s")
            time.sleep(retry_after)
            resp = self.session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise DiscordAPIError(resp.status_code, body)
        if resp.status_code == 204:
            return None
        return resp.json()

    def get_dm_channel(self, user_id: str) -> str:
        """Open/get a DM channel with a user. Returns channel ID."""
        data = self._request("POST", "/users/@me/channels", json={"recipient_id": user_id})
        return data["id"]

    def send_message(self, channel_id: str, content: str = None, embed: dict = None) -> dict:
        """Send a message to a channel. Returns message object."""
        payload = {}
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]
        return self._request("POST", f"/channels/{channel_id}/messages", json=payload)

    def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add a reaction to a message."""
        encoded = requests.utils.quote(emoji)
        self._request("PUT", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me")

    def get_reactions(self, channel_id: str, message_id: str, emoji: str) -> list:
        """Get users who reacted with a specific emoji."""
        encoded = requests.utils.quote(emoji)
        return self._request("GET", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}")

    def get_messages_after(self, channel_id: str, after_id: str, limit: int = 10) -> list:
        """Get messages in a channel after a specific message ID."""
        return self._request("GET", f"/channels/{channel_id}/messages", params={"after": after_id, "limit": limit})


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------

def _build_embed(
    title: str,
    description: str,
    level: str,
    event_type: str,
    context: dict,
) -> dict:
    """Build a Discord rich embed."""
    color = {
        INFO: COLOR_GREEN,
        APPROVAL_REQUIRED: COLOR_YELLOW,
        CRITICAL: COLOR_RED,
    }.get(level, COLOR_GREEN)

    fields = []
    for key, value in context.items():
        # Truncate long values
        val_str = str(value)
        if len(val_str) > 1024:
            val_str = val_str[:1021] + "..."
        fields.append({
            "name": str(key).replace("_", " ").title(),
            "value": f"```{val_str}```" if "\n" in val_str else val_str,
            "inline": len(val_str) < 40,
        })

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fields": fields,
        "footer": {
            "text": f"Savage22 | {level} | {event_type}",
        },
    }

    if level == APPROVAL_REQUIRED:
        embed["description"] += f"\n\nReact {REACTION_APPROVE} to approve or {REACTION_DENY} to deny.\nYou can also reply with **yes/no**."
    elif level == CRITICAL:
        embed["description"] += "\n\n**This alert will repeat every 5 minutes until acknowledged.**"

    return embed


# ---------------------------------------------------------------------------
# Gate Manager
# ---------------------------------------------------------------------------

class GateManager:
    """
    Discord-based notification and approval gate.

    Supports three levels:
    - INFO: fire-and-forget DM
    - APPROVAL_REQUIRED: blocks until user reacts or replies
    - CRITICAL: pings every 5 min until acknowledged
    """

    def __init__(self):
        self._client: Optional[DiscordClient] = None
        self._user_id: Optional[str] = None
        self._dm_channel_id: Optional[str] = None
        self._initialized = False
        self._init_lock = threading.Lock()
        # User overrides for event levels
        self._level_overrides: dict[str, str] = {}
        # Rate limiting (sliding window)
        self._rate_limit_timestamps: list[float] = []
        self._rate_limit_lock = threading.Lock()

    def _ensure_init(self) -> bool:
        """Lazy initialization from environment variables."""
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            token = os.environ.get("DISCORD_BOT_TOKEN")
            user_id = os.environ.get("DISCORD_USER_ID")
            if not token:
                logger.error("DISCORD_BOT_TOKEN not set. Gate manager disabled.")
                return False
            if not user_id:
                logger.error("DISCORD_USER_ID not set. Gate manager disabled.")
                return False
            try:
                self._client = DiscordClient(token)
                self._user_id = user_id
                self._dm_channel_id = self._client.get_dm_channel(user_id)
                self._initialized = True
                logger.info(f"Discord gate initialized. DM channel: {self._dm_channel_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Discord gate: {e}")
                return False

    def set_level(self, event_type: str, level: str) -> None:
        """Override the default notification level for an event type."""
        if level not in (INFO, APPROVAL_REQUIRED, CRITICAL):
            raise ValueError(f"Invalid level: {level}. Must be INFO, APPROVAL_REQUIRED, or CRITICAL")
        self._level_overrides[event_type] = level

    def _get_level(self, event_type: str, default: str) -> str:
        """Get the effective level for an event, considering overrides."""
        return self._level_overrides.get(event_type, default)

    def _get_priority_and_level(self, event_type: str) -> tuple[str, str]:
        """Get priority tier and notification level for an event type."""
        event_info = EVENT_DISPLAY.get(event_type)
        if event_info and len(event_info) >= 3:
            priority = event_info[2]
            level = PRIORITY_TO_LEVEL.get(priority, INFO)
        else:
            # Fallback for events not in EVENT_DISPLAY
            priority = P2_SUMMARY
            level = INFO
        # Apply user overrides
        level = self._level_overrides.get(event_type, level)
        return priority, level

    def _check_rate_limit(self, priority: str) -> bool:
        """
        Check if alert can be sent given rate limits.
        P0 (HALT) always allowed.
        P1/P2 rate-limited to 3 per 5 minutes.
        Returns True if allowed, False if rate-limited.
        """
        # P0 HALT always goes through
        if priority == P0_HALT:
            return True

        with self._rate_limit_lock:
            now = time.time()
            # Remove timestamps outside the window
            cutoff = now - RATE_LIMIT_WINDOW_S
            self._rate_limit_timestamps = [ts for ts in self._rate_limit_timestamps if ts > cutoff]

            # Check if under limit
            if len(self._rate_limit_timestamps) >= RATE_LIMIT_MAX:
                oldest = self._rate_limit_timestamps[0]
                wait_seconds = int((oldest + RATE_LIMIT_WINDOW_S) - now)
                logger.warning(f"Rate limit exceeded for {priority}. {len(self._rate_limit_timestamps)} alerts in last {RATE_LIMIT_WINDOW_S}s. Wait {wait_seconds}s.")
                return False

            # Add current timestamp
            self._rate_limit_timestamps.append(now)
            return True

    def _send_dm(self, content: str = None, embed: dict = None) -> Optional[dict]:
        """Send a DM to the user. Returns message dict or None."""
        if not self._ensure_init():
            return None
        try:
            return self._client.send_message(self._dm_channel_id, content=content, embed=embed)
        except DiscordAPIError as e:
            logger.error(f"Failed to send DM: {e}")
            return None

    def _add_approval_reactions(self, message_id: str) -> None:
        """Add approve/deny reactions to a message."""
        try:
            self._client.add_reaction(self._dm_channel_id, message_id, REACTION_APPROVE)
            time.sleep(0.3)  # Avoid rate limits
            self._client.add_reaction(self._dm_channel_id, message_id, REACTION_DENY)
        except DiscordAPIError as e:
            logger.error(f"Failed to add reactions: {e}")

    def _check_reaction(self, message_id: str) -> Optional[bool]:
        """Check if user reacted to the message. Returns True/False/None."""
        try:
            for emoji, result in [(REACTION_APPROVE, True), (REACTION_DENY, False)]:
                users = self._client.get_reactions(self._dm_channel_id, message_id, emoji)
                for user in users:
                    if str(user["id"]) == self._user_id:
                        return result
        except DiscordAPIError as e:
            logger.error(f"Failed to check reactions: {e}")
        return None

    def _check_text_reply(self, after_message_id: str) -> Optional[bool]:
        """Check if user sent a text reply after the gate message."""
        try:
            messages = self._client.get_messages_after(self._dm_channel_id, after_message_id)
            for msg in messages:
                if str(msg["author"]["id"]) != self._user_id:
                    continue
                text = msg["content"].strip().lower()
                if text in APPROVE_WORDS:
                    return True
                if text in DENY_WORDS:
                    return False
        except DiscordAPIError as e:
            logger.error(f"Failed to check text replies: {e}")
        return None

    def _poll_for_response(self, message_id: str, timeout_s: int = APPROVAL_ESCALATE_S) -> Optional[bool]:
        """
        Poll for user reaction or text reply.
        Returns True (approved), False (denied), or None (timeout/escalated).
        Sends a reminder at APPROVAL_TIMEOUT_S.
        """
        start = time.time()
        reminder_sent = False

        while time.time() - start < timeout_s:
            # Check reactions
            result = self._check_reaction(message_id)
            if result is not None:
                return result

            # Check text replies
            result = self._check_text_reply(message_id)
            if result is not None:
                return result

            # Send reminder at 30-minute mark
            elapsed = time.time() - start
            if elapsed >= APPROVAL_TIMEOUT_S and not reminder_sent:
                self._send_dm(content=f"**Reminder:** Approval still pending (waiting {int(elapsed / 60)} min). Please react or reply above.")
                reminder_sent = True
                logger.info("Sent approval reminder")

            time.sleep(POLL_INTERVAL_S)

        return None  # Timed out

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def notify(self, message: str, context: dict = None, event_type: str = "info") -> bool:
        """
        Send an INFO notification (non-blocking).

        Args:
            message: Human-readable summary
            context: Dict of key-value details
            event_type: Event category for logging/display

        Returns:
            True if sent successfully, False if rate-limited
        """
        context = context or {}
        priority, level = self._get_priority_and_level(event_type)

        # Check rate limit
        if not self._check_rate_limit(priority):
            logger.info(f"Rate-limited: {event_type} (priority {priority})")
            return False

        # If level was escalated, delegate
        if level == APPROVAL_REQUIRED:
            return self.approve(event_type, context, message=message)
        if level == CRITICAL:
            return self.critical(message, context, event_type=event_type)

        # Extract display info (handle both 2-tuple and 3-tuple)
        event_info = EVENT_DISPLAY.get(event_type)
        if event_info:
            display_name = event_info[0]
            display_desc = event_info[1] if len(event_info) > 1 else message
        else:
            display_name = message[:50]
            display_desc = message

        embed = _build_embed(
            title=display_name,
            description=message,
            level=INFO,
            event_type=event_type,
            context=context,
        )

        def _send():
            msg = self._send_dm(embed=embed)
            _append_log({
                "level": INFO,
                "priority": priority,
                "event_type": event_type,
                "message": message,
                "context": context,
                "sent": msg is not None,
                "message_id": msg["id"] if msg else None,
            })

        # Fire and forget in background thread
        t = threading.Thread(target=_send, daemon=True)
        t.start()
        return True

    def approve(self, event_type: str, context: dict = None, message: str = None) -> bool:
        """
        Send an APPROVAL_REQUIRED notification and block until response.

        Args:
            event_type: Event category (e.g. "rent_machine")
            context: Dict of details to display
            message: Optional override message

        Returns:
            True if approved, False if denied, timed out, or rate-limited
        """
        context = context or {}
        priority, level = self._get_priority_and_level(event_type)

        # Check rate limit
        if not self._check_rate_limit(priority):
            logger.warning(f"Rate-limited approval: {event_type} (priority {priority})")
            return False

        # If user downgraded to INFO, just notify and auto-approve
        if level == INFO:
            self.notify(message or f"Auto-approved: {event_type}", context, event_type=event_type)
            _append_log({
                "level": INFO,
                "priority": priority,
                "event_type": event_type,
                "message": f"Auto-approved (override): {event_type}",
                "context": context,
                "result": "auto_approved",
            })
            return True

        # Extract display info
        event_info = EVENT_DISPLAY.get(event_type)
        if event_info:
            display_name = event_info[0]
            display_desc = event_info[1] if len(event_info) > 1 else "Approval required"
        else:
            display_name = event_type.replace("_", " ").title()
            display_desc = "Approval required"

        if not message:
            message = display_desc

        embed = _build_embed(
            title=f"APPROVAL NEEDED: {display_name}",
            description=message,
            level=APPROVAL_REQUIRED,
            event_type=event_type,
            context=context,
        )

        msg = self._send_dm(embed=embed)
        if not msg:
            logger.error("Could not send approval request. Defaulting to DENIED.")
            _append_log({
                "level": APPROVAL_REQUIRED,
                "priority": priority,
                "event_type": event_type,
                "message": message,
                "context": context,
                "result": "send_failed",
            })
            return False

        message_id = msg["id"]
        self._add_approval_reactions(message_id)

        logger.info(f"Approval request sent for '{event_type}' (priority {priority}). Waiting for response...")
        result = self._poll_for_response(message_id)

        if result is None:
            # Escalate to CRITICAL
            logger.warning(f"Approval timeout for '{event_type}'. Escalating to CRITICAL.")
            self._send_dm(content=f"**ESCALATION:** No response for `{event_type}` after 2 hours. Treating as DENIED. Pinging CRITICAL.")
            self.critical(f"Unanswered approval: {event_type}", context, event_type=event_type)
            _append_log({
                "level": APPROVAL_REQUIRED,
                "priority": priority,
                "event_type": event_type,
                "message": message,
                "context": context,
                "result": "timeout_escalated",
                "message_id": message_id,
            })
            return False

        result_str = "approved" if result else "denied"
        logger.info(f"Approval for '{event_type}' (priority {priority}): {result_str}")

        # Confirm back to user
        confirm_emoji = REACTION_APPROVE if result else REACTION_DENY
        self._send_dm(content=f"{confirm_emoji} `{event_type}` — **{result_str.upper()}**. Proceeding.")

        _append_log({
            "level": APPROVAL_REQUIRED,
            "priority": priority,
            "event_type": event_type,
            "message": message,
            "context": context,
            "result": result_str,
            "message_id": message_id,
        })
        return result

    def critical(self, message: str, context: dict = None, event_type: str = "critical") -> bool:
        """
        Send a CRITICAL alert (P0 HALT). Pings every 5 minutes until acknowledged.
        P0 alerts bypass rate-limiting.

        Args:
            message: Human-readable summary
            context: Dict of details
            event_type: Event category

        Returns:
            True when acknowledged
        """
        context = context or {}
        priority, _ = self._get_priority_and_level(event_type)

        # P0 HALT always bypasses rate limits (already checked in _check_rate_limit)
        # But log it explicitly
        logger.warning(f"CRITICAL alert: {event_type} (priority {priority})")

        # Extract display info
        event_info = EVENT_DISPLAY.get(event_type)
        if event_info:
            display_name = event_info[0]
        else:
            display_name = "CRITICAL ALERT"

        embed = _build_embed(
            title=f"CRITICAL: {display_name}",
            description=message,
            level=CRITICAL,
            event_type=event_type,
            context=context,
        )

        msg = self._send_dm(content=f"**CRITICAL ALERT** <@{self._user_id}>", embed=embed)
        if not msg:
            logger.error("Could not send critical alert!")
            _append_log({
                "level": CRITICAL,
                "priority": priority,
                "event_type": event_type,
                "message": message,
                "context": context,
                "result": "send_failed",
            })
            return False

        message_id = msg["id"]
        self._add_approval_reactions(message_id)

        ping_count = 1
        start = time.time()
        last_ping = start

        while True:
            # Check for acknowledgment (any reaction or reply)
            react_result = self._check_reaction(message_id)
            if react_result is not None:
                break
            text_result = self._check_text_reply(message_id)
            if text_result is not None:
                break

            # Re-ping every 5 minutes
            if time.time() - last_ping >= CRITICAL_PING_INTERVAL_S:
                ping_count += 1
                self._send_dm(content=f"**CRITICAL PING #{ping_count}** <@{self._user_id}> — `{event_type}`: {message[:100]}")
                last_ping = time.time()
                logger.warning(f"Critical ping #{ping_count} for '{event_type}'")

            time.sleep(POLL_INTERVAL_S)

        elapsed = int(time.time() - start)
        self._send_dm(content=f"Critical alert `{event_type}` acknowledged after {elapsed}s ({ping_count} pings).")
        logger.info(f"Critical '{event_type}' (priority {priority}) acknowledged after {elapsed}s")

        _append_log({
            "level": CRITICAL,
            "priority": priority,
            "event_type": event_type,
            "message": message,
            "context": context,
            "result": "acknowledged",
            "message_id": message_id,
            "pings": ping_count,
            "elapsed_s": elapsed,
        })
        return True

    def gate_core_file(self, filepath: str, change_description: str) -> bool:
        """
        Check if a file is a core file and require approval if so.

        Args:
            filepath: Path to the file being changed
            change_description: What is being changed and why

        Returns:
            True if approved (or not a core file), False if denied
        """
        filename = Path(filepath).name
        if filename not in CORE_FILES:
            return True  # Not a core file, no gate needed

        return self.approve("core_file_change", {
            "file": filepath,
            "filename": filename,
            "change": change_description,
        }, message=f"Core file `{filename}` is being modified.")

    def gate_machine_rental(
        self,
        machine: str,
        cost_hr: str,
        est_hours: float,
        est_total: str,
        provider: str = "vast.ai",
        specs: str = "",
    ) -> bool:
        """
        Request approval to rent a cloud machine.

        Returns:
            True if approved, False if denied
        """
        return self.approve("rent_machine", {
            "machine": machine,
            "provider": provider,
            "cost_hr": cost_hr,
            "est_hours": est_hours,
            "est_total": est_total,
            "specs": specs,
        }, message=f"Requesting approval to rent **{machine}** on {provider}.")

    def gate_machine_destroy(self, machine: str, machine_id: str = "", reason: str = "") -> bool:
        """
        Request approval to destroy a cloud machine.

        Returns:
            True if approved, False if denied
        """
        return self.approve("destroy_machine", {
            "machine": machine,
            "machine_id": machine_id,
            "reason": reason,
        }, message=f"Requesting approval to **destroy** machine `{machine}`.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

gate = GateManager()

# ---------------------------------------------------------------------------
# CLI interface for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discord Gate Manager — test interface")
    sub = parser.add_subparsers(dest="command")

    # notify
    p_notify = sub.add_parser("notify", help="Send an INFO notification")
    p_notify.add_argument("message", help="Notification message")
    p_notify.add_argument("--event", default="info", help="Event type")
    p_notify.add_argument("--context", default="{}", help="JSON context dict")

    # approve
    p_approve = sub.add_parser("approve", help="Send an APPROVAL_REQUIRED request")
    p_approve.add_argument("event_type", help="Event type (e.g. rent_machine)")
    p_approve.add_argument("--message", default=None, help="Override message")
    p_approve.add_argument("--context", default="{}", help="JSON context dict")

    # critical
    p_critical = sub.add_parser("critical", help="Send a CRITICAL alert")
    p_critical.add_argument("message", help="Alert message")
    p_critical.add_argument("--event", default="critical", help="Event type")
    p_critical.add_argument("--context", default="{}", help="JSON context dict")

    # ping (test connectivity)
    sub.add_parser("ping", help="Test Discord connectivity")

    args = parser.parse_args()

    if args.command == "ping":
        if gate._ensure_init():
            msg = gate._send_dm(content="Savage22 Gate Manager -- connectivity test OK.")
            if msg:
                print(f"OK. Message ID: {msg['id']}")
            else:
                print("FAILED to send message.")
                sys.exit(1)
        else:
            print("FAILED to initialize. Check DISCORD_BOT_TOKEN and DISCORD_USER_ID.")
            sys.exit(1)

    elif args.command == "notify":
        ctx = json.loads(args.context)
        gate.notify(args.message, ctx, event_type=args.event)
        time.sleep(2)  # Wait for background thread
        print("Notification sent.")

    elif args.command == "approve":
        ctx = json.loads(args.context)
        result = gate.approve(args.event_type, ctx, message=args.message)
        print(f"Result: {'APPROVED' if result else 'DENIED'}")

    elif args.command == "critical":
        ctx = json.loads(args.context)
        gate.critical(args.message, ctx, event_type=args.event)
        print("Critical alert acknowledged.")

    else:
        parser.print_help()
