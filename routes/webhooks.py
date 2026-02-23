"""
Webhooks — Gmail Push Notifications (Pub/Sub)
When Gmail receives a new email it POSTs to this endpoint.
Setup: https://developers.google.com/gmail/api/guides/push
"""
from fastapi import APIRouter, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, User
from ai_agent import analyze_email
from gmail_service import GmailService
from routes.emails import save_email_analysis
import base64, json
from fastapi import Depends

router = APIRouter()


@router.post("/gmail")
async def gmail_push(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Gmail Pub/Sub push endpoint.
    Google sends a base64-encoded message containing {emailAddress, historyId}.
    """
    body = await request.json()
    message = body.get("message", {})
    data_b64 = message.get("data", "")

    try:
        decoded = base64.b64decode(data_b64 + "==").decode("utf-8")
        payload = json.loads(decoded)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub message")

    email_address = payload.get("emailAddress")
    history_id = payload.get("historyId")

    if not email_address:
        return {"status": "ignored"}

    # Find the user
    result = await db.execute(select(User).where(User.email == email_address))
    user = result.scalar_one_or_none()
    if not user or not user.access_token:
        return {"status": "user_not_found"}

    # Fetch and process new emails
    gmail = GmailService(user.access_token)
    try:
        raw_emails = await gmail.fetch_unread_emails(max_results=5)
        for raw in raw_emails:
            analysis = await analyze_email(
                subject=raw.get("subject", ""),
                body=raw.get("body", ""),
                sender=raw.get("sender", "")
            )
            if analysis["success"]:
                await save_email_analysis(db, user, raw, analysis)
                if raw.get("gmail_message_id"):
                    await gmail.mark_as_read(raw["gmail_message_id"])
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "processed"}


@router.post("/setup-watch")
async def setup_gmail_watch(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Setup Gmail push notifications (call once per user after OAuth).
    Requires a Google Pub/Sub topic configured.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/watch",
            headers={"Authorization": f"Bearer {user.access_token}"},
            json={
                "topicName": "projects/your-gcp-project/topics/mailbrain-gmail",
                "labelIds": ["INBOX"]
            }
        )
        return resp.json()
