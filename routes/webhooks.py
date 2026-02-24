import base64
import json
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, User

router = APIRouter()


@router.post("/gmail")
async def gmail_push(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        body    = await request.json()
        message = body.get("message", {})
        data    = base64.b64decode(message.get("data", "") + "==").decode("utf-8")
        payload = json.loads(data)
    except Exception:
        return {"status": "ignored"}

    email_address = payload.get("emailAddress")
    if not email_address:
        return {"status": "ignored"}

    result = await db.execute(select(User).where(User.email == email_address))
    user   = result.scalar_one_or_none()
    if not user or not user.access_token:
        return {"status": "user_not_found"}

    # Import here to avoid circular imports
    from routes.emails import gmail_request, parse_message, analyze_email, save_email_record

    try:
        data_resp = await gmail_request(user.access_token, "/users/me/messages",
                                        {"maxResults": 5, "q": "is:unread -from:me"})
        for msg_ref in data_resp.get("messages", []):
            raw      = await gmail_request(user.access_token, f"/users/me/messages/{msg_ref['id']}", {"format": "full"})
            parsed   = parse_message(raw)
            analysis = await analyze_email(parsed["subject"], parsed["body"], parsed["sender"])
            if analysis["success"]:
                await save_email_record(db, user, parsed, analysis)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "processed"}