"""
MailBrain API — Single file for Vercel deployment
All routes, models, and logic in one file to avoid import issues.
"""

import os
import base64
import uuid
import json
import re
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON, func, desc, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import select

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "https://06-mailbrain-api.vercel.app/auth/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "https://06-mailbrain.vercel.app")
JWT_SECRET           = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM        = "HS256"
JWT_EXPIRE_HOURS     = 24
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
AI_MODEL             = os.getenv("AI_MODEL", "gemini-2.5-flash")
AI_BASE_URL          = "https://generativelanguage.googleapis.com/v1beta/openai/"
AUTO_SEND_THRESHOLD  = float(os.getenv("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))
DATABASE_URL         = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./mailbrain.db")

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_API        = "https://gmail.googleapis.com/gmail/v1"

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────────────────────

connect_args = {}
if "neon.tech" in DATABASE_URL or ("postgresql" in DATABASE_URL and "aiosqlite" not in DATABASE_URL):
    connect_args = {"ssl": "require"}

engine = create_async_engine(DATABASE_URL, echo=False, connect_args=connect_args)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True)
    email         = Column(String, unique=True, nullable=False)
    name          = Column(String)
    picture       = Column(String)
    access_token  = Column(Text)
    refresh_token = Column(Text)
    token_expiry  = Column(DateTime)
    created_at    = Column(DateTime, default=datetime.utcnow)


class EmailModel(Base):
    __tablename__ = "emails"
    id                  = Column(String, primary_key=True)
    user_id             = Column(String, nullable=False)
    gmail_message_id    = Column(String)
    thread_id           = Column(String)
    sender              = Column(String)
    recipient           = Column(String)
    subject             = Column(String)
    body                = Column(Text)
    received_at         = Column(DateTime)
    intent              = Column(String)
    priority            = Column(String)
    priority_score      = Column(Float)
    sentiment           = Column(String)
    language            = Column(String, default="en")
    summary             = Column(Text)
    action_taken        = Column(String)
    assigned_department = Column(String)
    confidence_score    = Column(Float)
    generated_reply     = Column(Text)
    reply_sent          = Column(Boolean, default=False)
    reply_sent_at       = Column(DateTime)
    follow_up_at        = Column(DateTime)
    escalated           = Column(Boolean, default=False)
    raw_headers         = Column(JSON)
    ai_metadata         = Column(JSON)
    processed_at        = Column(DateTime, default=datetime.utcnow)
    status              = Column(String, default="processed")


class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String)
    email_id    = Column(String)
    action      = Column(String)
    details     = Column(JSON)
    created_at  = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ──────────────────────────────────────────────────────────────────────────────
# APP STARTUP
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
        print("✅ DB ready")
    except Exception as e:
        print(f"❌ DB error: {e}")
        traceback.print_exc()
    yield


app = FastAPI(title="MailBrain API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(UserModel).where(UserModel.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ──────────────────────────────────────────────────────────────────────────────
# GMAIL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

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


def parse_gmail_message(raw: dict) -> dict:
    import email as email_lib
    headers = {h["name"].lower(): h["value"] for h in raw.get("payload", {}).get("headers", [])}
    body = extract_body(raw.get("payload", {}))
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
        "body": body,
        "received_at": received_at,
        "raw_headers": dict(headers),
    }


async def gmail_get(access_token: str, path: str, params: dict = None):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GMAIL_API}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params or {},
            timeout=30,
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
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GMAIL_API}/users/me/messages/send",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

# ──────────────────────────────────────────────────────────────────────────────
# AI ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

async def analyze_email(subject: str, body: str, sender: str, thread_context: str = None) -> dict:
    prompt = f"""Analyze this email and respond ONLY with valid JSON (no markdown, no backticks):

FROM: {sender}
SUBJECT: {subject}
BODY:
{body[:3000]}
{"THREAD CONTEXT: " + thread_context[:1000] if thread_context else ""}

Return exactly this JSON structure:
{{
  "intent": "one of: support_request | refund_demand | sales_inquiry | meeting_request | complaint | spam | urgent_escalation | billing_question | partnership_offer | general_inquiry",
  "priority": "one of: CRITICAL | HIGH | NORMAL | LOW",
  "priority_score": 0.0,
  "sentiment": "one of: positive | neutral | negative",
  "language": "ISO 639-1 code like en, es, fr",
  "summary": "one sentence summary",
  "action": "one of: auto_reply | assign_department | create_ticket | schedule_meeting | flag_management | request_info",
  "assigned_department": "one of: support | billing | sales | management | technical | null",
  "confidence_score": 0.0,
  "escalation_risk": false,
  "follow_up_needed": false,
  "follow_up_hours": null,
  "reply_tone": "one of: professional | empathetic | friendly | firm",
  "generated_reply": "complete ready-to-send reply email text",
  "keywords_detected": []
}}"""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{AI_BASE_URL}chat/completions",
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": AI_MODEL,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"^```\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            analysis = json.loads(raw_text)
            return {"success": True, "data": analysis}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# SAVE EMAIL TO DB
# ──────────────────────────────────────────────────────────────────────────────

async def save_email(db: AsyncSession, user: UserModel, raw: dict, analysis: dict) -> EmailModel:
    d = analysis.get("data", {})
    follow_up_at = None
    if d.get("follow_up_needed") and d.get("follow_up_hours"):
        follow_up_at = datetime.utcnow() + timedelta(hours=d["follow_up_hours"])

    record = EmailModel(
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

    # Auto-send if confidence high enough
    if (
        d.get("action") == "auto_reply"
        and float(d.get("confidence_score", 0)) >= AUTO_SEND_THRESHOLD
        and user.access_token
        and raw.get("sender")
    ):
        try:
            await gmail_send(user.access_token, raw["sender"], raw.get("subject", ""), d.get("generated_reply", ""), raw.get("thread_id"))
            record.reply_sent = True
            record.reply_sent_at = datetime.utcnow()
        except Exception as e:
            record.status = f"reply_failed: {str(e)[:100]}"

    db.add(record)
    log = ActivityLog(user_id=user.id, email_id=record.id, action=d.get("action", "processed"),
                      details={"intent": d.get("intent"), "priority": d.get("priority"), "auto_sent": record.reply_sent})
    db.add(log)
    await db.commit()
    await db.refresh(record)
    return record

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — BASE
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "MailBrain API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — AUTH
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/auth/google")
async def auth_google():
    scopes = "%20".join(GMAIL_SCOPES)
    url = (
        f"{GOOGLE_AUTH_URL}"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scopes}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url)


@app.get("/auth/callback")
async def auth_callback(code: str, db: AsyncSession = Depends(get_db)):
    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }, timeout=30)

        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp.text}")

        tokens = token_resp.json()

        user_resp = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10,
        )
        user_info = user_resp.json()

    # Upsert user
    result = await db.execute(select(UserModel).where(UserModel.email == user_info["email"]))
    user = result.scalar_one_or_none()
    expiry = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))

    if not user:
        user = UserModel(
            id=str(uuid.uuid4()),
            email=user_info["email"],
            name=user_info.get("name"),
            picture=user_info.get("picture"),
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_expiry=expiry,
        )
        db.add(user)
    else:
        user.access_token  = tokens.get("access_token")
        user.refresh_token = tokens.get("refresh_token", user.refresh_token)
        user.token_expiry  = expiry
        user.name          = user_info.get("name")
        user.picture       = user_info.get("picture")

    await db.commit()
    await db.refresh(user)

    token = create_jwt(user.id, user.email)
    return RedirectResponse(f"{FRONTEND_URL}/dashboard?token={token}")


@app.get("/auth/me")
async def auth_me(user: UserModel = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "picture": user.picture}


@app.post("/auth/logout")
async def auth_logout():
    return {"message": "Logged out"}

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — EMAILS
# ──────────────────────────────────────────────────────────────────────────────

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


@app.get("/emails/")
async def list_emails(
    page: int = 1, page_size: int = 20,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(EmailModel).where(EmailModel.user_id == user.id).order_by(desc(EmailModel.processed_at))
    if intent:   q = q.where(EmailModel.intent == intent)
    if priority: q = q.where(EmailModel.priority == priority)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows  = (await db.execute(q.offset((page - 1) * page_size).limit(page_size))).scalars().all()

    return {
        "total": total, "page": page, "page_size": page_size,
        "emails": [
            {
                "id": e.id, "sender": e.sender, "subject": e.subject, "summary": e.summary,
                "intent": e.intent, "priority": e.priority, "priority_score": e.priority_score,
                "sentiment": e.sentiment, "action_taken": e.action_taken,
                "confidence_score": e.confidence_score, "reply_sent": e.reply_sent,
                "escalated": e.escalated, "received_at": str(e.received_at),
                "processed_at": str(e.processed_at), "status": e.status,
            }
            for e in rows
        ],
    }


@app.post("/emails/sync")
async def sync_emails(
    max_results: int = 20,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not user.access_token:
        raise HTTPException(status_code=400, detail="No Gmail access token. Please reconnect Gmail.")

    try:
        data = await gmail_get(user.access_token, "/users/me/messages", {"maxResults": max_results, "q": "is:unread -from:me"})
        messages = data.get("messages", [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gmail fetch failed: {str(e)}")

    if not messages:
        return {"message": "No new emails", "processed": 0}

    processed, errors = [], []
    for msg_ref in messages:
        try:
            raw_msg = await gmail_get(user.access_token, f"/users/me/messages/{msg_ref['id']}", {"format": "full"})
            parsed  = parse_gmail_message(raw_msg)
            analysis = await analyze_email(parsed["subject"], parsed["body"], parsed["sender"])
            if analysis["success"]:
                saved = await save_email(db, user, parsed, analysis)
                processed.append({"id": saved.id, "subject": saved.subject, "intent": saved.intent, "priority": saved.priority})
                # Mark as read
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{GMAIL_API}/users/me/messages/{msg_ref['id']}/modify",
                        headers={"Authorization": f"Bearer {user.access_token}"},
                        json={"removeLabelIds": ["UNREAD"]},
                        timeout=10,
                    )
            else:
                errors.append({"id": msg_ref["id"], "error": analysis.get("error")})
        except Exception as e:
            errors.append({"id": msg_ref.get("id"), "error": str(e)})

    return {"processed": len(processed), "errors": len(errors), "emails": processed}


@app.post("/emails/process")
async def process_email(
    data: ManualEmailInput,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    analysis = await analyze_email(data.subject, data.body, data.sender, data.thread_context)
    if not analysis["success"]:
        raise HTTPException(status_code=500, detail=f"AI error: {analysis.get('error')}")

    raw = {"sender": data.sender, "subject": data.subject, "body": data.body,
           "received_at": datetime.utcnow(), "raw_headers": {}}
    saved = await save_email(db, user, raw, analysis)

    return {"id": saved.id, "analysis": analysis["data"], "reply_sent": saved.reply_sent}


@app.post("/emails/batch")
async def batch_process(
    batch: BatchInput,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for e in batch.emails:
        analysis = await analyze_email(e.subject, e.body, e.sender, e.thread_context)
        if analysis["success"]:
            raw = {"sender": e.sender, "subject": e.subject, "body": e.body,
                   "received_at": datetime.utcnow(), "raw_headers": {}}
            saved = await save_email(db, user, raw, analysis)
            results.append({"id": saved.id, "intent": saved.intent, "priority": saved.priority})
    return {"processed": len(results), "emails": results}


@app.get("/emails/{email_id}")
async def get_email(
    email_id: str,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(EmailModel).where(EmailModel.id == email_id, EmailModel.user_id == user.id))
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


@app.post("/emails/{email_id}/approve")
async def approve_reply(
    email_id: str,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(EmailModel).where(EmailModel.id == email_id, EmailModel.user_id == user.id))
    email = result.scalar_one_or_none()
    if not email:    raise HTTPException(status_code=404, detail="Email not found")
    if email.reply_sent: raise HTTPException(status_code=400, detail="Reply already sent")
    if not email.generated_reply: raise HTTPException(status_code=400, detail="No reply to send")

    await gmail_send(user.access_token, email.sender, email.subject, email.generated_reply, email.thread_id)
    email.reply_sent    = True
    email.reply_sent_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}


@app.post("/emails/{email_id}/reply")
async def send_reply(
    email_id: str,
    reply: ReplyInput,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(EmailModel).where(EmailModel.id == email_id, EmailModel.user_id == user.id))
    email = result.scalar_one_or_none()
    if not email: raise HTTPException(status_code=404, detail="Email not found")

    await gmail_send(user.access_token, email.sender, email.subject, reply.body, email.thread_id)
    email.reply_sent    = True
    email.reply_sent_at = datetime.utcnow()
    email.generated_reply = reply.body
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/analytics/overview")
async def analytics_overview(
    days: int = 7,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    base  = select(EmailModel).where(EmailModel.user_id == user.id, EmailModel.processed_at >= since)

    total      = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    critical   = (await db.execute(select(func.count()).select_from(base.where(EmailModel.priority == "CRITICAL").subquery()))).scalar() or 0
    auto_sent  = (await db.execute(select(func.count()).select_from(base.where(EmailModel.reply_sent == True).subquery()))).scalar() or 0
    escalated  = (await db.execute(select(func.count()).select_from(base.where(EmailModel.escalated == True).subquery()))).scalar() or 0
    avg_conf   = (await db.execute(select(func.avg(EmailModel.confidence_score)).where(EmailModel.user_id == user.id, EmailModel.processed_at >= since))).scalar() or 0

    return {
        "period_days": days, "total_emails": total, "critical_emails": critical,
        "auto_replied": auto_sent, "escalated": escalated,
        "automation_rate": round(auto_sent / total * 100 if total else 0, 1),
        "avg_confidence": round(float(avg_conf) * 100, 1),
    }


@app.get("/analytics/intent")
async def analytics_intent(
    days: int = 30,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailModel.intent, func.count(EmailModel.id).label("count"))
        .where(EmailModel.user_id == user.id, EmailModel.processed_at >= since)
        .group_by(EmailModel.intent).order_by(desc("count"))
    )).all()
    return {"distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]}


@app.get("/analytics/priority")
async def analytics_priority(
    days: int = 7,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailModel.priority, func.count(EmailModel.id).label("count"))
        .where(EmailModel.user_id == user.id, EmailModel.processed_at >= since)
        .group_by(EmailModel.priority)
    )).all()
    return {"breakdown": [{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows]}


@app.get("/analytics/trends")
async def analytics_trends(
    days: int = 14,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(
            func.date(EmailModel.processed_at).label("date"),
            func.count(EmailModel.id).label("total"),
        )
        .where(EmailModel.user_id == user.id, EmailModel.processed_at >= since)
        .group_by(func.date(EmailModel.processed_at))
        .order_by(func.date(EmailModel.processed_at))
    )).all()
    return {"trends": [{"date": str(r.date), "total": r.total} for r in rows]}


@app.get("/analytics/automation")
async def analytics_automation(
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(EmailModel.action_taken, func.count(EmailModel.id).label("count"))
        .where(EmailModel.user_id == user.id)
        .group_by(EmailModel.action_taken).order_by(desc("count"))
    )).all()
    total = sum(r.count for r in rows)
    return {
        "total_processed": total,
        "actions": [{"action": r.action_taken or "none", "count": r.count,
                     "percentage": round(r.count / total * 100, 1) if total else 0} for r in rows],
    }


@app.get("/analytics/escalations")
async def analytics_escalations(
    days: int = 7,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailModel)
        .where(EmailModel.user_id == user.id, EmailModel.processed_at >= since,
               (EmailModel.escalated == True) | (EmailModel.priority == "CRITICAL"))
        .order_by(desc(EmailModel.priority_score))
    )).scalars().all()
    return {
        "count": len(rows),
        "emails": [{"id": e.id, "sender": e.sender, "subject": e.subject,
                    "priority": e.priority, "escalated": e.escalated, "reply_sent": e.reply_sent} for e in rows],
    }