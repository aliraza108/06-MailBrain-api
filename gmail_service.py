"""
Gmail API integration — fetch, parse, and send emails.
"""
import base64
import email as email_lib
from email.mime.text import MIMEText
from typing import Optional
from datetime import datetime
import httpx
from config import get_settings

settings = get_settings()

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"


class GmailService:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {"Authorization": f"Bearer {access_token}"}

    # ──────────────────────────────────────────
    # Fetch Emails
    # ──────────────────────────────────────────

    async def list_messages(self, max_results: int = 20, query: str = "is:unread") -> list[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GMAIL_API}/users/me/messages",
                headers=self.headers,
                params={"maxResults": max_results, "q": query}
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("messages", [])

    async def get_message(self, message_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GMAIL_API}/users/me/messages/{message_id}",
                headers=self.headers,
                params={"format": "full"}
            )
            resp.raise_for_status()
            return resp.json()

    async def get_thread(self, thread_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GMAIL_API}/users/me/threads/{thread_id}",
                headers=self.headers
            )
            resp.raise_for_status()
            return resp.json()

    # ──────────────────────────────────────────
    # Parse Email
    # ──────────────────────────────────────────

    def parse_message(self, raw_message: dict) -> dict:
        headers = {h["name"].lower(): h["value"] for h in raw_message.get("payload", {}).get("headers", [])}

        body = self._extract_body(raw_message.get("payload", {}))

        # Parse date
        date_str = headers.get("date", "")
        try:
            received_at = email_lib.utils.parsedate_to_datetime(date_str)
        except Exception:
            received_at = datetime.utcnow()

        return {
            "gmail_message_id": raw_message.get("id"),
            "thread_id": raw_message.get("threadId"),
            "sender": headers.get("from", ""),
            "recipient": headers.get("to", ""),
            "subject": headers.get("subject", "(no subject)"),
            "body": body,
            "received_at": received_at.isoformat(),
            "raw_headers": dict(headers),
        }

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract plain text body from Gmail payload."""
        mime_type = payload.get("mimeType", "")

        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")

        if mime_type.startswith("multipart/"):
            for part in payload.get("parts", []):
                body = self._extract_body(part)
                if body:
                    return body

        return ""

    def _extract_sender_name(self, from_header: str) -> str:
        """Extract name from 'Name <email@domain.com>' format."""
        if "<" in from_header:
            return from_header.split("<")[0].strip().strip('"')
        return from_header.split("@")[0]

    def _extract_sender_email(self, from_header: str) -> str:
        if "<" in from_header:
            return from_header.split("<")[1].rstrip(">")
        return from_header

    # ──────────────────────────────────────────
    # Send Email
    # ──────────────────────────────────────────

    async def send_reply(self, to: str, subject: str, body: str, thread_id: Optional[str] = None) -> dict:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        payload = {"raw": raw}
        if thread_id:
            payload["threadId"] = thread_id

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GMAIL_API}/users/me/messages/send",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload
            )
            resp.raise_for_status()
            return resp.json()

    async def mark_as_read(self, message_id: str):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{GMAIL_API}/users/me/messages/{message_id}/modify",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"removeLabelIds": ["UNREAD"]}
            )

    async def add_label(self, message_id: str, label_name: str):
        # Get or create label first
        label_id = await self._get_or_create_label(label_name)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{GMAIL_API}/users/me/messages/{message_id}/modify",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"addLabelIds": [label_id]}
            )

    async def _get_or_create_label(self, label_name: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{GMAIL_API}/users/me/labels", headers=self.headers)
            labels = resp.json().get("labels", [])
            for label in labels:
                if label["name"].lower() == label_name.lower():
                    return label["id"]

            # Create it
            resp = await client.post(
                f"{GMAIL_API}/users/me/labels",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
            )
            return resp.json()["id"]

    # ──────────────────────────────────────────
    # Fetch + Parse in One Call
    # ──────────────────────────────────────────

    async def fetch_unread_emails(self, max_results: int = 20) -> list[dict]:
        messages = await self.list_messages(max_results=max_results, query="is:unread -from:me")
        parsed = []
        for msg_ref in messages:
            raw = await self.get_message(msg_ref["id"])
            parsed.append(self.parse_message(raw))
        return parsed

    async def get_thread_context(self, thread_id: str, current_msg_id: str) -> str:
        """Get previous messages in thread as context string."""
        thread = await self.get_thread(thread_id)
        messages = thread.get("messages", [])
        context_parts = []
        for msg in messages:
            if msg.get("id") == current_msg_id:
                break
            parsed = self.parse_message(msg)
            context_parts.append(f"FROM: {parsed['sender']}\nSUBJECT: {parsed['subject']}\n{parsed['body'][:500]}")
        return "\n---\n".join(context_parts[-3:])  # Last 3 messages for context
