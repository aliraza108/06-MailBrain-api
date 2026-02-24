import base64
import json

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import User, get_db

router = APIRouter()


@router.post("/gmail")
async def gmail_webhook(
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """Receive Gmail Pub/Sub push notifications for real-time processing."""
    try:
        body    = await request.json()
        message = body.get("message", {})
        raw     = base64.b64decode(message.get("data", "") + "==").decode()
        payload = json.loads(raw)
    except Exception:
        return {"status": "ignored"}

    email_address = payload.get("emailAddress")
    if not email_address:
        return {"status": "ignored"}

    result = await db.execute(select(User).where(User.email == email_address))
    user   = result.scalar_one_or_none()
    if not user or not user.access_token:
        return {"status": "user_not_found"}

    # Late imports to avoid circular dependency at module load time
    from ai_agent import analyze_email
    from gmail_service import GmailService
    from routes.emails import save_email_to_db

    gmail = GmailService(user.access_token)

    try:
        raw_emails = await gmail.fetch_and_parse_unread(max_results=5)
        for raw_email in raw_emails:
            ctx = None
            if raw_email.get("thread_id") and raw_email.get("gmail_message_id"):
                ctx = await gmail.get_thread_context(
                    raw_email["thread_id"], raw_email["gmail_message_id"]
                ) or None
            analysis = await analyze_email(
                subject        = raw_email.get("subject", ""),
                body           = raw_email.get("body", ""),
                sender         = raw_email.get("sender", ""),
                thread_context = ctx,
            )
            saved = await save_email_to_db(db, user, raw_email, analysis)
            if saved and raw_email.get("gmail_message_id"):
                await gmail.mark_as_read(raw_email["gmail_message_id"])
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "processed"}