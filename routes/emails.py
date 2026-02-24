import uuid
import httpx
import base64
import json
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel

from database import get_db, User, Email, ActivityLog
from config import GEMINI_API_KEY, AI_MODEL, AI_BASE_URL, AUTO_SEND_THRESHOLD
from routes.auth import verify_jwt

router = APIRouter()

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"


# ── Auth helper ────────────────────────────────────────────────────────────────

async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ── Schemas ────────────────────────────────────────────────────────────────────

class ManualEmailInput(BaseModel):
    sender: str
    sender_name: Optional[str] = ""
    subject: str
    body: str
    thread_context: Optional[str] = None

class BatchInput(BaseModel):
    emails: list[ManualEmailInput]

class ReplyInput(BaseModel):
    body: str


# ── Gmail helpers ──────────────────────────────────────────────────────────────

def extract_body(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            body = extract_body(part)
            if body:
                return body
    return ""


def parse_message(raw: dict) -> dict:
    import email as email_lib
    headers = {h["name"].lower(): h["value"] for h in raw.get("payload", {}).get("headers", [])}
    try:
        received_at = email_lib.utils.parsedate_to_datetime(headers.get("date", ""))
    except Exception:
        received_at = datetime.utcnow()
    return {
        "gmail_message_id": raw.get("id"),
        "thread_id": raw.get("threadId"),
        "sender": headers.get("from", ""),
        "recipient": headers.get("to", ""),
        "subject": headers.get("subject", "(no subject)"),
        "body": extract_body(raw.get("payload", {})),
        "received_at": received_at,
        "raw_headers": dict(headers),
    }


async def gmail_request(access_token: str, path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GMAIL_API}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()


async def gmail_send(access_token: str, to: str, subject: str, body: str, thread_id: str = None):
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GMAIL_API}/users/me/messages/send",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── AI Analysis ────────────────────────────────────────────────────────────────

async def analyze_email(subject: str, body: str, sender: str, thread_context: str = None) -> dict:
    prompt = f"""Analyze this email and respond ONLY with valid JSON. No markdown, no backticks, just JSON.

FROM: {sender}
SUBJECT: {subject}
BODY:
{body[:3000]}
{"THREAD CONTEXT:\\n" + thread_context[:1000] if thread_context else ""}

Return this exact JSON:
{{
  "intent": "support_request|refund_demand|sales_inquiry|meeting_request|complaint|spam|urgent_escalation|billing_question|partnership_offer|general_inquiry",
  "priority": "CRITICAL|HIGH|NORMAL|LOW",
  "priority_score": 0.8,
  "sentiment": "positive|neutral|negative",
  "language": "en",
  "summary": "one sentence",
  "action": "auto_reply|assign_department|create_ticket|schedule_meeting|flag_management|request_info",
  "assigned_department": "support|billing|sales|management|technical|null",
  "confidence_score": 0.9,
  "escalation_risk": false,
  "follow_up_needed": false,
  "follow_up_hours": null,
  "reply_tone": "professional|empathetic|friendly|firm",
  "generated_reply": "Dear ..., full reply text here",
  "keywords_detected": ["keyword1", "keyword2"]
}}"""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{AI_BASE_URL}chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
                json={"model": AI_MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            return {"success": True, "data": json.loads(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Save to DB ─────────────────────────────────────────────────────────────────

async def save_email_record(db: AsyncSession, user: User, raw: dict, analysis: dict) -> Email:
    d = analysis.get("data", {})
    follow_up_at = None
    if d.get("follow_up_needed") and d.get("follow_up_hours"):
        follow_up_at = datetime.utcnow() + timedelta(hours=float(d["follow_up_hours"]))

    record = Email(
        id=str(uuid.uuid4()),
        user_id=user.id,
        gmail_message_id=raw.get("gmail_message_id"),
        thread_id=raw.get("thread_id"),
        sender=raw.get("sender", ""),
        recipient=raw.get("recipient", ""),
        subject=raw.get("subject", ""),
        body=raw.get("body", ""),
        received_at=raw.get("received_at"),
        intent=d.get("intent"),
        priority=d.get("priority"),
        priority_score=d.get("priority_score"),
        sentiment=d.get("sentiment"),
        language=d.get("language", "en"),
        summary=d.get("summary"),
        action_taken=d.get("action"),
        assigned_department=d.get("assigned_department"),
        confidence_score=d.get("confidence_score"),
        generated_reply=d.get("generated_reply"),
        escalated=d.get("escalation_risk", False),
        follow_up_at=follow_up_at,
        ai_metadata={"keywords": d.get("keywords_detected", []), "tone": d.get("reply_tone")},
        raw_headers=raw.get("raw_headers", {}),
        status="processed",
    )

    # Auto-send if confident enough
    if (
        d.get("action") == "auto_reply"
        and float(d.get("confidence_score") or 0) >= AUTO_SEND_THRESHOLD
        and user.access_token
        and raw.get("sender")
    ):
        try:
            await gmail_send(user.access_token, raw["sender"], raw.get("subject", ""),
                             d.get("generated_reply", ""), raw.get("thread_id"))
            record.reply_sent    = True
            record.reply_sent_at = datetime.utcnow()
        except Exception as e:
            record.status = f"reply_failed: {str(e)[:100]}"

    db.add(record)
    db.add(ActivityLog(
        user_id=user.id, email_id=record.id, action=d.get("action", "processed"),
        details={"intent": d.get("intent"), "priority": d.get("priority"), "auto_sent": record.reply_sent}
    ))
    await db.commit()
    await db.refresh(record)
    return record


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/")
async def list_emails(
    page: int = 1, page_size: int = 20,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(Email).where(Email.user_id == user.id).order_by(desc(Email.processed_at))
    if intent:   q = q.where(Email.intent == intent)
    if priority: q = q.where(Email.priority == priority)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows  = (await db.execute(q.offset((page - 1) * page_size).limit(page_size))).scalars().all()

    return {
        "total": total, "page": page, "page_size": page_size,
        "emails": [{
            "id": e.id, "sender": e.sender, "subject": e.subject, "summary": e.summary,
            "intent": e.intent, "priority": e.priority, "priority_score": e.priority_score,
            "sentiment": e.sentiment, "action_taken": e.action_taken,
            "confidence_score": e.confidence_score, "reply_sent": e.reply_sent,
            "escalated": e.escalated, "received_at": str(e.received_at),
            "processed_at": str(e.processed_at), "status": e.status,
        } for e in rows],
    }


@router.post("/sync")
async def sync_emails(
    max_results: int = 20,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not user.access_token:
        raise HTTPException(status_code=400, detail="No Gmail token. Please reconnect Gmail.")
    try:
        data = await gmail_request(user.access_token, "/users/me/messages",
                                   {"maxResults": max_results, "q": "is:unread -from:me"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gmail fetch failed: {str(e)}")

    messages = data.get("messages", [])
    if not messages:
        return {"message": "No new emails", "processed": 0}

    processed, errors = [], []
    for msg_ref in messages:
        try:
            raw_msg  = await gmail_request(user.access_token, f"/users/me/messages/{msg_ref['id']}", {"format": "full"})
            parsed   = parse_message(raw_msg)
            analysis = await analyze_email(parsed["subject"], parsed["body"], parsed["sender"])
            if analysis["success"]:
                saved = await save_email_record(db, user, parsed, analysis)
                processed.append({"id": saved.id, "subject": saved.subject, "intent": saved.intent, "priority": saved.priority})
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(f"{GMAIL_API}/users/me/messages/{msg_ref['id']}/modify",
                                      headers={"Authorization": f"Bearer {user.access_token}"},
                                      json={"removeLabelIds": ["UNREAD"]})
            else:
                errors.append({"error": analysis.get("error")})
        except Exception as e:
            errors.append({"error": str(e)})

    return {"processed": len(processed), "errors": len(errors), "emails": processed}


@router.post("/process")
async def process_email(
    data: ManualEmailInput,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    analysis = await analyze_email(data.subject, data.body, data.sender, data.thread_context)
    if not analysis["success"]:
        raise HTTPException(status_code=500, detail=f"AI error: {analysis.get('error')}")
    raw = {"sender": data.sender, "subject": data.subject, "body": data.body,
           "received_at": datetime.utcnow(), "raw_headers": {}}
    saved = await save_email_record(db, user, raw, analysis)
    return {"id": saved.id, "analysis": analysis["data"], "reply_sent": saved.reply_sent}


@router.post("/batch")
async def batch_process(
    batch: BatchInput,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for e in batch.emails:
        analysis = await analyze_email(e.subject, e.body, e.sender, e.thread_context)
        if analysis["success"]:
            raw = {"sender": e.sender, "subject": e.subject, "body": e.body,
                   "received_at": datetime.utcnow(), "raw_headers": {}}
            saved = await save_email_record(db, user, raw, analysis)
            results.append({"id": saved.id, "intent": saved.intent, "priority": saved.priority})
    return {"processed": len(results), "emails": results}


@router.get("/{email_id}")
async def get_email(
    email_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Email).where(Email.id == email_id, Email.user_id == user.id))
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return {
        "id": email.id, "sender": email.sender, "recipient": email.recipient,
        "subject": email.subject, "body": email.body, "received_at": str(email.received_at),
        "intent": email.intent, "priority": email.priority, "priority_score": email.priority_score,
        "sentiment": email.sentiment, "language": email.language, "summary": email.summary,
        "action_taken": email.action_taken, "assigned_department": email.assigned_department,
        "confidence_score": email.confidence_score, "generated_reply": email.generated_reply,
        "reply_sent": email.reply_sent, "reply_sent_at": str(email.reply_sent_at),
        "escalated": email.escalated, "ai_metadata": email.ai_metadata, "status": email.status,
    }


@router.post("/{email_id}/approve")
async def approve_reply(
    email_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Email).where(Email.id == email_id, Email.user_id == user.id))
    email = result.scalar_one_or_none()
    if not email:          raise HTTPException(status_code=404, detail="Email not found")
    if email.reply_sent:   raise HTTPException(status_code=400, detail="Reply already sent")
    if not email.generated_reply: raise HTTPException(status_code=400, detail="No generated reply")

    await gmail_send(user.access_token, email.sender, email.subject, email.generated_reply, email.thread_id)
    email.reply_sent    = True
    email.reply_sent_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}


@router.post("/{email_id}/reply")
async def send_reply(
    email_id: str,
    reply: ReplyInput,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Email).where(Email.id == email_id, Email.user_id == user.id))
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    await gmail_send(user.access_token, email.sender, email.subject, reply.body, email.thread_id)
    email.reply_sent      = True
    email.reply_sent_at   = datetime.utcnow()
    email.generated_reply = reply.body
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}