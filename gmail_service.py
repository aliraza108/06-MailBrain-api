"""
Gmail API service.
Handles fetching, parsing, sending, and labelling emails.
"""
import base64
import email as stdlib_email
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx
from config import GMAIL_API_BASE


class GmailService:
    def __init__(self, access_token: str):
        self.token   = access_token
        self._headers = {"Authorization": f"Bearer {access_token}"}

    # ── Low-level requests ────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict = None) -> dict:
        url = f"{GMAIL_API_BASE}{path}"
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url, headers=self._headers, params=params or {})
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, body: dict) -> dict:
        url = f"{GMAIL_API_BASE}{path}"
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                url,
                headers={**self._headers, "Content-Type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            return r.json()

    # ── List & fetch ──────────────────────────────────────────────────────────

    async def list_unread(self, max_results: int = 20) -> list[dict]:
        """Return list of {id, threadId} stubs for unread inbox messages."""
        try:
            data = await self._get("/users/me/messages", {
                "maxResults": max_results,
                "q":          "is:unread in:inbox -from:me",
            })
            return data.get("messages", [])
        except Exception:
            return []

    async def get_message(self, message_id: str) -> Optional[dict]:
        """Fetch full message by ID."""
        try:
            return await self._get(f"/users/me/messages/{message_id}", {"format": "full"})
        except Exception:
            return None

    async def get_thread_messages(self, thread_id: str) -> list[dict]:
        """Return all messages in a thread."""
        try:
            data = await self._get(f"/users/me/threads/{thread_id}")
            return data.get("messages", [])
        except Exception:
            return []

    # ── Parse ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_body(payload: dict) -> str:
        """Recursively extract plain-text body from Gmail message payload."""
        mime = payload.get("mimeType", "")

        if mime == "text/plain":
            raw = payload.get("body", {}).get("data", "")
            if raw:
                try:
                    return base64.urlsafe_b64decode(raw + "==").decode("utf-8", errors="ignore")
                except Exception:
                    return ""

        if mime.startswith("multipart/"):
            # Prefer text/plain parts
            for part in payload.get("parts", []):
                if part.get("mimeType") == "text/plain":
                    result = GmailService._decode_body(part)
                    if result:
                        return result
            # Fall back to any part
            for part in payload.get("parts", []):
                result = GmailService._decode_body(part)
                if result:
                    return result

        return ""

    def parse_message(self, raw: dict) -> dict:
        """Convert raw Gmail API response into clean dict."""
        if not raw:
            return {}

        hdrs = {
            h["name"].lower(): h["value"]
            for h in raw.get("payload", {}).get("headers", [])
        }

        # Parse received date
        try:
            received_at = stdlib_email.utils.parsedate_to_datetime(hdrs.get("date", ""))
        except Exception:
            received_at = datetime.utcnow()

        body = self._decode_body(raw.get("payload", {}))

        return {
            "gmail_message_id": raw.get("id"),
            "thread_id":        raw.get("threadId"),
            "sender":           hdrs.get("from", ""),
            "recipient":        hdrs.get("to", ""),
            "subject":          hdrs.get("subject", "(no subject)"),
            "body":             body,
            "received_at":      received_at,
            "raw_headers":      hdrs,
        }

    # ── Thread context ────────────────────────────────────────────────────────

    async def get_thread_context(self, thread_id: str, skip_message_id: str) -> str:
        """
        Build a context string from previous messages in a thread.
        Used to give AI conversation history.
        """
        messages = await self.get_thread_messages(thread_id)
        parts = []
        for msg in messages:
            if msg.get("id") == skip_message_id:
                continue
            parsed = self.parse_message(msg)
            snippet = (parsed.get("body") or "")[:400]
            parts.append(
                f"FROM: {parsed.get('sender', '')}\n"
                f"SUBJECT: {parsed.get('subject', '')}\n"
                f"{snippet}"
            )
        # Return last 3 messages as context
        return "\n---\n".join(parts[-3:]) if parts else ""

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send_reply(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Send a reply via Gmail API."""
        # Build MIME message
        msg = MIMEText(body, "plain", "utf-8")
        msg["To"]      = to
        msg["Subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"

        raw_bytes = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        payload: dict = {"raw": raw_bytes}
        if thread_id:
            payload["threadId"] = thread_id

        return await self._post("/users/me/messages/send", payload)

    # ── Modify ────────────────────────────────────────────────────────────────

    async def mark_as_read(self, message_id: str) -> None:
        """Remove UNREAD label."""
        try:
            await self._post(
                f"/users/me/messages/{message_id}/modify",
                {"removeLabelIds": ["UNREAD"]},
            )
        except Exception:
            pass  # Non-critical — don't crash if this fails

    async def apply_label(self, message_id: str, label_name: str) -> None:
        """Apply a named label to a message, creating the label if needed."""
        try:
            label_id = await self._get_or_create_label(label_name)
            await self._post(
                f"/users/me/messages/{message_id}/modify",
                {"addLabelIds": [label_id]},
            )
        except Exception:
            pass

    async def _get_or_create_label(self, name: str) -> str:
        data = await self._get("/users/me/labels")
        for label in data.get("labels", []):
            if label.get("name", "").lower() == name.lower():
                return label["id"]
        # Create new label
        new_label = await self._post("/users/me/labels", {
            "name": name,
            "labelListVisibility":   "labelShow",
            "messageListVisibility": "show",
        })
        return new_label["id"]

    # ── High-level fetch ──────────────────────────────────────────────────────

    async def fetch_and_parse_unread(self, max_results: int = 20) -> list[dict]:
        """
        Fetch unread messages and return list of parsed dicts.
        Skips any messages that fail to fetch.
        """
        stubs = await self.list_unread(max_results)
        results = []
        for stub in stubs:
            raw = await self.get_message(stub["id"])
            if raw:
                parsed = self.parse_message(raw)
                if parsed:
                    results.append(parsed)
        return results