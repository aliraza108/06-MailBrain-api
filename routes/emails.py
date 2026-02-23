"""
Email Routes — Core MailBrain Processing Pipeline
GET  /emails/              List processed emails
POST /emails/sync          Fetch & process new Gmail emails
POST /emails/process       Process uploaded/pasted email (manual)
GET  /emails/{id}          Get single email detail
POST /emails/{id}/reply    Send reply for email
POST /emails/{id}/approve  Approve & send AI-generated reply
POST /emails/batch         Process multiple emails
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import uuid

from database import get_db, Email, User, ActivityLog
from ai_agent import analyze_email, analyze_batch
from gmail_service import GmailService
from routes.auth import verify_jwt
from config import get_settings

settings = get_settings()
router = APIRouter()


# ─────────────────────────────────────────────
# Auth helper
# ─────────────────────────────────────────────

async def get_user_from_token(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class ManualEmailInput(BaseModel):
    sender: str
    sender_name: Optional[str] = ""
    subject: str
    body: str
    thread_context: Optional[str] = None

class BatchEmailInput(BaseModel):
    emails: list[ManualEmailInput]

class ReplyInput(BaseModel):
    body: str
    send_immediately: bool = False


# ─────────────────────────────────────────────
# Internal: Save analyzed email to DB
# ─────────────────────────────────────────────

async def save_email_analysis(
    db: AsyncSession,
    user: User,
    raw_email: dict,
    analysis: dict
) -> Email:
    analysis_data = analysis.get("data", {})

    follow_up_at = None
    if analysis_data.get("follow_up_needed") and analysis_data.get("follow_up_hours"):
        follow_up_at = datetime.utcnow() + timedelta(hours=analysis_data["follow_up_hours"])

    email_record = Email(
        id=str(uuid.uuid4()),
        user_id=user.id,
        gmail_message_id=raw_email.get("gmail_message_id"),
        thread_id=raw_email.get("thread_id"),
        sender=raw_email.get("sender", ""),
        recipient=raw_email.get("recipient", ""),
        subject=raw_email.get("subject", ""),
        body=raw_email.get("body", ""),
        received_at=raw_email.get("received_at"),
        intent=analysis_data.get("intent"),
        priority=analysis_data.get("priority"),
        priority_score=analysis_data.get("priority_score"),
        sentiment=analysis_data.get("sentiment"),
        language=analysis_data.get("language", "en"),
        summary=analysis_data.get("summary"),
        action_taken=analysis_data.get("action"),
        assigned_department=analysis_data.get("assigned_department"),
        confidence_score=analysis_data.get("confidence_score"),
        generated_reply=analysis_data.get("generated_reply"),
        escalated=analysis_data.get("escalation_risk", False),
        follow_up_at=follow_up_at,
        ai_metadata={
            "keywords_detected": analysis_data.get("keywords_detected", []),
            "reply_tone": analysis_data.get("reply_tone"),
        },
        raw_headers=raw_email.get("raw_headers", {}),
        status="processed"
    )

    # Auto-send if confidence is high enough and action is auto_reply
    if (
        analysis_data.get("action") == "auto_reply"
        and analysis_data.get("confidence_score", 0) >= settings.AUTO_SEND_CONFIDENCE_THRESHOLD
        and user.access_token
    ):
        try:
            gmail = GmailService(user.access_token)
            await gmail.send_reply(
                to=raw_email.get("sender", ""),
                subject=raw_email.get("subject", ""),
                body=analysis_data.get("generated_reply", ""),
                thread_id=raw_email.get("thread_id")
            )
            email_record.reply_sent = True
            email_record.reply_sent_at = datetime.utcnow()
        except Exception as e:
            email_record.status = f"reply_failed: {str(e)}"

    db.add(email_record)

    # Log the activity
    log = ActivityLog(
        user_id=user.id,
        email_id=email_record.id,
        action=analysis_data.get("action", "processed"),
        details={
            "intent": analysis_data.get("intent"),
            "priority": analysis_data.get("priority"),
            "confidence": analysis_data.get("confidence_score"),
            "auto_sent": email_record.reply_sent
        }
    )
    db.add(log)
    await db.commit()
    await db.refresh(email_record)
    return email_record


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get("/")
async def list_emails(
    page: int = 1,
    page_size: int = 20,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """List all processed emails with optional filters."""
    query = select(Email).where(Email.user_id == user.id).order_by(desc(Email.processed_at))

    if intent:
        query = query.where(Email.intent == intent)
    if priority:
        query = query.where(Email.priority == priority)
    if status:
        query = query.where(Email.status == status)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar()

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    emails = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "emails": [
            {
                "id": e.id,
                "sender": e.sender,
                "subject": e.subject,
                "summary": e.summary,
                "intent": e.intent,
                "priority": e.priority,
                "priority_score": e.priority_score,
                "sentiment": e.sentiment,
                "language": e.language,
                "action_taken": e.action_taken,
                "assigned_department": e.assigned_department,
                "confidence_score": e.confidence_score,
                "reply_sent": e.reply_sent,
                "escalated": e.escalated,
                "received_at": e.received_at,
                "processed_at": e.processed_at,
                "status": e.status
            }
            for e in emails
        ]
    }


@router.post("/sync")
async def sync_gmail(
    max_results: int = 20,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Fetch unread Gmail emails and run full MailBrain analysis on each."""
    if not user.access_token:
        raise HTTPException(status_code=400, detail="No Gmail access token. Please reconnect Gmail.")

    gmail = GmailService(user.access_token)

    try:
        raw_emails = await gmail.fetch_unread_emails(max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gmail fetch failed: {str(e)}")

    if not raw_emails:
        return {"message": "No new emails found", "processed": 0}

    processed = []
    errors = []

    for raw in raw_emails:
        # Get thread context if in thread
        thread_context = None
        if raw.get("thread_id"):
            try:
                thread_context = await gmail.get_thread_context(
                    raw["thread_id"], raw.get("gmail_message_id", "")
                )
            except Exception:
                pass

        # Run AI analysis
        analysis = await analyze_email(
            subject=raw.get("subject", ""),
            body=raw.get("body", ""),
            sender=raw.get("sender", ""),
            thread_context=thread_context
        )

        if analysis["success"]:
            saved = await save_email_analysis(db, user, raw, analysis)
            processed.append({
                "id": saved.id,
                "subject": saved.subject,
                "intent": saved.intent,
                "priority": saved.priority,
                "action": saved.action_taken,
                "reply_sent": saved.reply_sent
            })
            # Mark as read in Gmail
            if raw.get("gmail_message_id"):
                try:
                    await gmail.mark_as_read(raw["gmail_message_id"])
                except Exception:
                    pass
        else:
            errors.append({"subject": raw.get("subject"), "error": analysis.get("error")})

    return {
        "processed": len(processed),
        "errors": len(errors),
        "emails": processed,
        "error_details": errors
    }


@router.post("/process")
async def process_manual_email(
    email_input: ManualEmailInput,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Process a manually entered or pasted email."""
    analysis = await analyze_email(
        subject=email_input.subject,
        body=email_input.body,
        sender=email_input.sender,
        sender_name=email_input.sender_name or "",
        thread_context=email_input.thread_context
    )

    if not analysis["success"]:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {analysis.get('error')}")

    raw_email = {
        "sender": email_input.sender,
        "recipient": "",
        "subject": email_input.subject,
        "body": email_input.body,
        "received_at": datetime.utcnow().isoformat(),
        "raw_headers": {}
    }

    saved = await save_email_analysis(db, user, raw_email, analysis)

    return {
        "id": saved.id,
        "analysis": analysis["data"],
        "reply_sent": saved.reply_sent,
        "status": saved.status
    }


@router.post("/batch")
async def process_batch(
    batch: BatchEmailInput,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Process multiple emails at once."""
    emails_data = [
        {
            "subject": e.subject,
            "body": e.body,
            "sender": e.sender,
            "sender_name": e.sender_name or "",
            "thread_context": e.thread_context
        }
        for e in batch.emails
    ]

    results = await analyze_batch(emails_data)
    saved_ids = []

    for i, (email_input, analysis) in enumerate(zip(batch.emails, results)):
        if analysis["success"]:
            raw_email = {
                "sender": email_input.sender,
                "subject": email_input.subject,
                "body": email_input.body,
                "received_at": datetime.utcnow().isoformat(),
                "raw_headers": {}
            }
            saved = await save_email_analysis(db, user, raw_email, analysis)
            saved_ids.append({"index": i, "id": saved.id, "intent": saved.intent, "priority": saved.priority})

    return {"processed": len(saved_ids), "emails": saved_ids}


@router.get("/{email_id}")
async def get_email(
    email_id: str,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Get full details of a single email."""
    result = await db.execute(
        select(Email).where(Email.id == email_id, Email.user_id == user.id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    return {
        "id": email.id,
        "sender": email.sender,
        "recipient": email.recipient,
        "subject": email.subject,
        "body": email.body,
        "received_at": email.received_at,
        "processed_at": email.processed_at,
        "intent": email.intent,
        "priority": email.priority,
        "priority_score": email.priority_score,
        "sentiment": email.sentiment,
        "language": email.language,
        "summary": email.summary,
        "action_taken": email.action_taken,
        "assigned_department": email.assigned_department,
        "confidence_score": email.confidence_score,
        "generated_reply": email.generated_reply,
        "reply_sent": email.reply_sent,
        "reply_sent_at": email.reply_sent_at,
        "escalated": email.escalated,
        "follow_up_at": email.follow_up_at,
        "ai_metadata": email.ai_metadata,
        "status": email.status
    }


@router.post("/{email_id}/approve")
async def approve_and_send_reply(
    email_id: str,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Approve the AI-generated reply and send it via Gmail."""
    result = await db.execute(
        select(Email).where(Email.id == email_id, Email.user_id == user.id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    if email.reply_sent:
        raise HTTPException(status_code=400, detail="Reply already sent")
    if not email.generated_reply:
        raise HTTPException(status_code=400, detail="No generated reply to send")
    if not user.access_token:
        raise HTTPException(status_code=400, detail="No Gmail access token")

    gmail = GmailService(user.access_token)
    try:
        await gmail.send_reply(
            to=email.sender,
            subject=email.subject,
            body=email.generated_reply,
            thread_id=email.thread_id
        )
        email.reply_sent = True
        email.reply_sent_at = datetime.utcnow()
        await db.commit()
        return {"message": "Reply sent successfully", "email_id": email_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send reply: {str(e)}")


@router.post("/{email_id}/reply")
async def send_custom_reply(
    email_id: str,
    reply: ReplyInput,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Send a custom reply for an email."""
    result = await db.execute(
        select(Email).where(Email.id == email_id, Email.user_id == user.id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    if not user.access_token:
        raise HTTPException(status_code=400, detail="No Gmail access token")

    gmail = GmailService(user.access_token)
    await gmail.send_reply(
        to=email.sender,
        subject=email.subject,
        body=reply.body,
        thread_id=email.thread_id
    )
    email.reply_sent = True
    email.reply_sent_at = datetime.utcnow()
    email.generated_reply = reply.body
    await db.commit()

    return {"message": "Reply sent", "email_id": email_id}
