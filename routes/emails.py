import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ai_agent import analyze_email
from config import AUTO_SEND_THRESHOLD
from database import ActivityLog, EmailRecord, User, get_db
from gmail_service import GmailService
from routes.auth import decode_token

router = APIRouter()


# ── Auth dependency ───────────────────────────────────────────────────────────

async def require_user(
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <token>")
    payload = decode_token(authorization[7:].strip())
    result  = await db.execute(select(User).where(User.id == payload["sub"]))
    user    = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return user


# ── Schemas ───────────────────────────────────────────────────────────────────

class EmailInput(BaseModel):
    sender:         str
    sender_name:    Optional[str] = ""
    subject:        str
    body:           str
    thread_context: Optional[str] = None


class BatchIn(BaseModel):
    emails: list[EmailInput]


class ReplyIn(BaseModel):
    body: str


# ── Core: save analysis to DB ─────────────────────────────────────────────────

async def save_email_to_db(
    db:       AsyncSession,
    user:     User,
    raw:      dict,
    analysis: dict,
) -> EmailRecord:
    """
    Save an AI-analysed email to the database.
    Auto-sends reply if confidence >= threshold and action is auto_reply.
    Skips duplicate gmail_message_id.
    """
    d = analysis.get("data", {})

    # Skip duplicates (same Gmail message already processed)
    gmail_id = raw.get("gmail_message_id")
    if gmail_id:
        existing = await db.execute(
            select(EmailRecord).where(EmailRecord.gmail_message_id == gmail_id)
        )
        if existing.scalar_one_or_none():
            return None  # Already processed

    # Compute follow-up time
    follow_up_at = None
    if d.get("follow_up_needed") and d.get("follow_up_hours"):
        follow_up_at = datetime.utcnow() + timedelta(hours=float(d["follow_up_hours"]))

    record = EmailRecord(
        id                  = str(uuid.uuid4()),
        user_id             = user.id,
        gmail_message_id    = gmail_id,
        thread_id           = raw.get("thread_id"),
        sender              = raw.get("sender", ""),
        recipient           = raw.get("recipient", ""),
        subject             = raw.get("subject", ""),
        body                = raw.get("body", ""),
        received_at         = raw.get("received_at"),
        intent              = d.get("intent"),
        priority            = d.get("priority"),
        priority_score      = d.get("priority_score"),
        sentiment           = d.get("sentiment"),
        language            = d.get("language", "en"),
        summary             = d.get("summary"),
        action_taken        = d.get("action"),
        assigned_department = d.get("assigned_department"),
        confidence_score    = d.get("confidence_score"),
        generated_reply     = d.get("generated_reply"),
        escalated           = bool(d.get("escalation_risk", False)),
        follow_up_at        = follow_up_at,
        ai_metadata         = {
            "keywords":    d.get("keywords_detected", []),
            "tone":        d.get("reply_tone"),
            "ai_success":  analysis.get("success", False),
            "ai_error":    analysis.get("error"),
        },
        raw_headers         = raw.get("raw_headers", {}),
        status              = "processed",
    )

    # Auto-send reply if confidence is high enough
    if (
        d.get("action") == "auto_reply"
        and float(d.get("confidence_score") or 0) >= AUTO_SEND_THRESHOLD
        and user.access_token
        and raw.get("sender")
    ):
        try:
            gmail = GmailService(user.access_token)
            await gmail.send_reply(
                to        = raw["sender"],
                subject   = raw.get("subject", ""),
                body      = d.get("generated_reply", ""),
                thread_id = raw.get("thread_id"),
            )
            record.reply_sent    = True
            record.reply_sent_at = datetime.utcnow()
        except Exception as e:
            record.status = f"reply_failed: {str(e)[:120]}"

    db.add(record)
    db.add(ActivityLog(
        user_id  = user.id,
        email_id = record.id,
        action   = d.get("action", "processed"),
        details  = {
            "intent":     d.get("intent"),
            "priority":   d.get("priority"),
            "confidence": d.get("confidence_score"),
            "auto_sent":  record.reply_sent,
            "ai_success": analysis.get("success"),
        },
    ))

    await db.commit()
    await db.refresh(record)
    return record


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/")
async def list_emails(
    page:      int           = 1,
    page_size: int           = 20,
    intent:    Optional[str] = None,
    priority:  Optional[str] = None,
    user:      User          = Depends(require_user),
    db:        AsyncSession  = Depends(get_db),
):
    q = (
        select(EmailRecord)
        .where(EmailRecord.user_id == user.id)
        .order_by(desc(EmailRecord.processed_at))
    )
    if intent:   q = q.where(EmailRecord.intent == intent)
    if priority: q = q.where(EmailRecord.priority == priority)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows  = (await db.execute(q.offset((page - 1) * page_size).limit(page_size))).scalars().all()

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "emails": [{
            "id":               e.id,
            "sender":           e.sender,
            "subject":          e.subject,
            "summary":          e.summary,
            "intent":           e.intent,
            "priority":         e.priority,
            "priority_score":   e.priority_score,
            "sentiment":        e.sentiment,
            "action_taken":     e.action_taken,
            "assigned_department": e.assigned_department,
            "confidence_score": e.confidence_score,
            "reply_sent":       e.reply_sent,
            "escalated":        e.escalated,
            "received_at":      str(e.received_at),
            "processed_at":     str(e.processed_at),
            "status":           e.status,
        } for e in rows],
    }


@router.post("/sync")
async def sync_gmail(
    max_results: int          = 20,
    user:        User         = Depends(require_user),
    db:          AsyncSession = Depends(get_db),
):
    """
    Pull unread Gmail messages, run AI analysis on each,
    save to database, mark as read.
    """
    if not user.access_token:
        raise HTTPException(400, "No Gmail token. Visit /auth/google to reconnect.")

    gmail = GmailService(user.access_token)

    # Fetch unread emails
    try:
        raw_emails = await gmail.fetch_and_parse_unread(max_results)
    except Exception as e:
        raise HTTPException(400, f"Gmail fetch failed: {str(e)}")

    if not raw_emails:
        return {"message": "No unread emails found", "processed": 0, "emails": []}

    processed, skipped, errors = [], [], []

    for raw in raw_emails:
        try:
            # Get thread context for better AI understanding
            thread_ctx = None
            if raw.get("thread_id") and raw.get("gmail_message_id"):
                thread_ctx = await gmail.get_thread_context(
                    raw["thread_id"], raw["gmail_message_id"]
                ) or None

            # Run AI analysis
            analysis = await analyze_email(
                subject        = raw.get("subject", ""),
                body           = raw.get("body", ""),
                sender         = raw.get("sender", ""),
                thread_context = thread_ctx,
            )

            # Save to DB
            saved = await save_email_to_db(db, user, raw, analysis)

            if saved is None:
                skipped.append(raw.get("subject", "?"))
                continue

            processed.append({
                "id":         saved.id,
                "subject":    saved.subject,
                "sender":     saved.sender,
                "intent":     saved.intent,
                "priority":   saved.priority,
                "action":     saved.action_taken,
                "reply_sent": saved.reply_sent,
                "ai_success": analysis.get("success"),
            })

            # Mark as read in Gmail
            if raw.get("gmail_message_id"):
                await gmail.mark_as_read(raw["gmail_message_id"])

        except Exception as e:
            errors.append({
                "subject": raw.get("subject", "?"),
                "error":   str(e),
            })

    return {
        "processed": len(processed),
        "skipped":   len(skipped),
        "errors":    len(errors),
        "emails":    processed,
        "error_details": errors,
    }


@router.post("/process")
async def process_manual(
    data: EmailInput,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    """Manually submit an email for AI analysis and save result."""
    analysis = await analyze_email(
        subject        = data.subject,
        body           = data.body,
        sender         = data.sender,
        thread_context = data.thread_context,
    )
    raw = {
        "sender":      data.sender,
        "subject":     data.subject,
        "body":        data.body,
        "received_at": datetime.utcnow(),
        "raw_headers": {},
    }
    saved = await save_email_to_db(db, user, raw, analysis)
    return {
        "id":         saved.id,
        "analysis":   analysis["data"],
        "reply_sent": saved.reply_sent,
        "status":     saved.status,
        "ai_success": analysis.get("success"),
    }


@router.post("/batch")
async def batch_process(
    batch: BatchIn,
    user:  User         = Depends(require_user),
    db:    AsyncSession = Depends(get_db),
):
    """Process a batch of emails at once."""
    results = []
    for e in batch.emails:
        analysis = await analyze_email(e.subject, e.body, e.sender, e.thread_context)
        raw = {
            "sender":      e.sender,
            "subject":     e.subject,
            "body":        e.body,
            "received_at": datetime.utcnow(),
            "raw_headers": {},
        }
        saved = await save_email_to_db(db, user, raw, analysis)
        if saved:
            results.append({
                "id":       saved.id,
                "intent":   saved.intent,
                "priority": saved.priority,
            })
    return {"processed": len(results), "emails": results}


@router.get("/{email_id}")
async def get_email(
    email_id: str,
    user:     User         = Depends(require_user),
    db:       AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmailRecord).where(
            EmailRecord.id      == email_id,
            EmailRecord.user_id == user.id,
        )
    )
    e = result.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    return {
        "id":                  e.id,
        "sender":              e.sender,
        "recipient":           e.recipient,
        "subject":             e.subject,
        "body":                e.body,
        "received_at":         str(e.received_at),
        "intent":              e.intent,
        "priority":            e.priority,
        "priority_score":      e.priority_score,
        "sentiment":           e.sentiment,
        "language":            e.language,
        "summary":             e.summary,
        "action_taken":        e.action_taken,
        "assigned_department": e.assigned_department,
        "confidence_score":    e.confidence_score,
        "generated_reply":     e.generated_reply,
        "reply_sent":          e.reply_sent,
        "reply_sent_at":       str(e.reply_sent_at),
        "escalated":           e.escalated,
        "follow_up_at":        str(e.follow_up_at),
        "ai_metadata":         e.ai_metadata,
        "status":              e.status,
    }


@router.post("/{email_id}/approve")
async def approve_reply(
    email_id: str,
    user:     User         = Depends(require_user),
    db:       AsyncSession = Depends(get_db),
):
    """Send the AI-generated reply for an email."""
    result = await db.execute(
        select(EmailRecord).where(
            EmailRecord.id      == email_id,
            EmailRecord.user_id == user.id,
        )
    )
    e = result.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    if e.reply_sent:
        raise HTTPException(400, "Reply already sent")
    if not e.generated_reply:
        raise HTTPException(400, "No AI reply available — process the email first")
    if not user.access_token:
        raise HTTPException(400, "No Gmail token — reconnect at /auth/google")

    gmail = GmailService(user.access_token)
    await gmail.send_reply(e.sender, e.subject, e.generated_reply, e.thread_id)

    e.reply_sent    = True
    e.reply_sent_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reply sent successfully", "email_id": email_id}


@router.post("/{email_id}/reply")
async def send_custom_reply(
    email_id: str,
    reply:    ReplyIn,
    user:     User         = Depends(require_user),
    db:       AsyncSession = Depends(get_db),
):
    """Send a custom reply for an email."""
    result = await db.execute(
        select(EmailRecord).where(
            EmailRecord.id      == email_id,
            EmailRecord.user_id == user.id,
        )
    )
    e = result.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    if not user.access_token:
        raise HTTPException(400, "No Gmail token — reconnect at /auth/google")

    gmail = GmailService(user.access_token)
    await gmail.send_reply(e.sender, e.subject, reply.body, e.thread_id)

    e.reply_sent      = True
    e.reply_sent_at   = datetime.utcnow()
    e.generated_reply = reply.body
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}