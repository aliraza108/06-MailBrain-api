"""
MailBrain API - single file for Vercel serverless
"""
import os, sys, uuid, base64, json, re, traceback, httpx, jwt
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional
import email as stdlib_email

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import (Column, String, Integer, DateTime, Text, Float,
                        Boolean, JSON, func, desc, select, text)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# Section 2: Configuration
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "https://06-mailbrain-api.vercel.app/auth/callback")
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "https://06-mailbrain.vercel.app")
JWT_SECRET           = os.environ.get("JWT_SECRET", "dev-secret")
JWT_ALGO             = "HS256"
JWT_HOURS            = 24
GEMINI_KEY           = os.environ.get("GEMINI_API_KEY", "")
AI_MODEL             = os.environ.get("AI_MODEL", "gemini-2.5-flash")
AI_URL               = "https://generativelanguage.googleapis.com/v1beta/openai/"
AUTO_THRESHOLD       = float(os.environ.get("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))
DATABASE_URL         = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./mailbrain.db")

# Auto-fix DATABASE_URL for asyncpg
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_BASE       = "https://gmail.googleapis.com/gmail/v1"
SCOPES = (
    "openid "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile "
    "https://www.googleapis.com/auth/gmail.readonly "
    "https://www.googleapis.com/auth/gmail.send "
    "https://www.googleapis.com/auth/gmail.modify"
)

# Section 3: Database Models
_ssl = {"ssl": "require"} if "postgresql" in DATABASE_URL else {}
engine = create_async_engine(DATABASE_URL, echo=False, connect_args=_ssl)
SessionFactory = sessionmaker(engine, class_=AsyncSession,
                              expire_on_commit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True)
    email         = Column(String, unique=True, nullable=False)
    name          = Column(String)
    picture       = Column(String)
    access_token  = Column(Text)
    refresh_token = Column(Text)
    token_expiry  = Column(DateTime)
    created_at    = Column(DateTime, default=datetime.utcnow)


class EmailRecord(Base):
    __tablename__ = "emails"
    id                  = Column(String, primary_key=True)
    user_id             = Column(String, nullable=False, index=True)
    gmail_message_id    = Column(String, unique=True, nullable=True)
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
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String, nullable=False)
    email_id   = Column(String)
    action     = Column(String)
    details    = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

async def get_db():
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Section 4: App + Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[MB] DB tables ready")
    except Exception as e:
        print(f"[MB] DB error: {e}")
        traceback.print_exc()
    yield

app = FastAPI(title="MailBrain API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})

# Section 5: Auth Helpers

def make_token(user_id: str, email: str) -> str:
    return jwt.encode(
        {"sub": user_id, "email": email,
         "exp": datetime.utcnow() + timedelta(hours=JWT_HOURS),
         "iat": datetime.utcnow()},
        JWT_SECRET, algorithm=JWT_ALGO)


def read_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except Exception:
        raise HTTPException(401, "Invalid token")


async def auth(authorization: str = Header(None),
               db: AsyncSession = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization: Bearer <token> required")
    payload = read_token(authorization[7:].strip())
    res = await db.execute(select(User).where(User.id == payload["sub"]))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return user

# Section 6: AI Analysis

def _rule_based(subject: str, body: str) -> dict:
    t = (subject + " " + body).lower()
    if any(w in t for w in ["urgent", "asap", "emergency", "immediately", "critical"]):
        intent, priority, dept, score = "urgent_escalation", "CRITICAL", "management", 0.95
    elif any(w in t for w in ["refund", "money back", "reimburse", "chargeback"]):
        intent, priority, dept, score = "refund_demand", "HIGH", "billing", 0.80
    elif any(w in t for w in ["complaint", "unacceptable", "terrible", "disgusting"]):
        intent, priority, dept, score = "complaint", "HIGH", "support", 0.75
    elif any(w in t for w in ["invoice", "billing", "payment", "subscription"]):
        intent, priority, dept, score = "billing_question", "HIGH", "billing", 0.75
    elif any(w in t for w in ["meeting", "schedule", "call", "demo", "appointment"]):
        intent, priority, dept, score = "meeting_request", "NORMAL", "sales", 0.70
    elif any(w in t for w in ["partner", "collaborat", "integrate", "business opportunity"]):
        intent, priority, dept, score = "partnership_offer", "LOW", "sales", 0.60
    elif any(w in t for w in ["unsubscribe", "spam", "promotional"]):
        intent, priority, dept, score = "spam", "LOW", "none", 0.50
    elif any(w in t for w in ["bug", "error", "broken", "not working", "crash"]):
        intent, priority, dept, score = "support_request", "HIGH", "technical", 0.75
    else:
        intent, priority, dept, score = "general_inquiry", "NORMAL", "support", 0.60
    neg = any(w in t for w in ["angry", "furious", "terrible", "awful", "hate", "worst"])
    return {
        "intent": intent,
        "priority": priority,
        "priority_score": score,
        "sentiment": "negative" if neg else "neutral",
        "language": "en",
        "summary": f"Email about: {subject[:100]}",
        "action": "assign_department",
        "assigned_department": dept,
        "confidence_score": 0.60,
        "escalation_risk": priority in ("CRITICAL", "HIGH"),
        "follow_up_needed": priority == "CRITICAL",
        "follow_up_hours": 2 if priority == "CRITICAL" else None,
        "reply_tone": "empathetic" if neg else "professional",
        "generated_reply": (
            f"Dear Customer,\n\nThank you for contacting us regarding \"{subject}\".\n"
            f"Our {dept} team will respond within 24 hours.\n\nBest regards,\nMailBrain Support"
        ),
        "keywords_detected": [],
    }


async def ai_analyze(subject: str, body: str, sender: str, thread_ctx: str = None) -> dict:
    if not GEMINI_KEY:
        return {"success": False, "data": _rule_based(subject, body)}

    thread_part = f"\nPREVIOUS THREAD:\n{thread_ctx[:600]}" if thread_ctx else ""
    prompt = (
        "Analyze this email. Return ONLY valid JSON. No markdown. No backticks.\n\n"
        f"FROM: {sender}\n"
        f"SUBJECT: {subject}\n"
        f"BODY: {(body or '')[:3000]}{thread_part}\n\n"
        "JSON (all fields required):\n"
        "{\n"
        "  \"intent\": \"support_request|refund_demand|sales_inquiry|meeting_request|complaint|spam|urgent_escalation|billing_question|partnership_offer|general_inquiry\",\n"
        "  \"priority\": \"CRITICAL|HIGH|NORMAL|LOW\",\n"
        "  \"priority_score\": 0.85,\n"
        "  \"sentiment\": \"positive|neutral|negative\",\n"
        "  \"language\": \"en\",\n"
        "  \"summary\": \"One sentence summary.\",\n"
        "  \"action\": \"auto_reply|assign_department|create_ticket|schedule_meeting|flag_management|request_info\",\n"
        "  \"assigned_department\": \"support|billing|sales|management|technical|none\",\n"
        "  \"confidence_score\": 0.90,\n"
        "  \"escalation_risk\": false,\n"
        "  \"follow_up_needed\": false,\n"
        "  \"follow_up_hours\": null,\n"
        "  \"reply_tone\": \"professional|empathetic|friendly|firm\",\n"
        "  \"generated_reply\": \"Dear Customer, full reply text...\",\n"
        "  \"keywords_detected\": []\n"
        "}"
    )

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                f"{AI_URL}chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": AI_MODEL,
                    "max_tokens": 2048,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON. No markdown. No backticks."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        data = json.loads(raw)

        defaults = {
            "intent": "general_inquiry",
            "priority": "NORMAL",
            "priority_score": 0.5,
            "sentiment": "neutral",
            "language": "en",
            "summary": "Email received.",
            "action": "auto_reply",
            "assigned_department": "support",
            "confidence_score": 0.7,
            "escalation_risk": False,
            "follow_up_needed": False,
            "follow_up_hours": None,
            "reply_tone": "professional",
            "generated_reply": "Thank you for your email. We will respond shortly.",
            "keywords_detected": [],
        }
        for k, v in defaults.items():
            if k not in data or data[k] is None:
                data[k] = v
        return {"success": True, "data": data}
    except Exception:
        return {"success": False, "data": _rule_based(subject, body)}

# Section 7: Gmail Helpers

def _b64_decode(data: str) -> str:
    if not data:
        return ""
    try:
        pad = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + pad).decode(errors="ignore")
    except Exception:
        return ""


def _extract_body(payload) -> str:
    try:
        if not payload:
            return ""
        mime = payload.get("mimeType", "")
        body = payload.get("body", {}) or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            return _b64_decode(data)
        if mime.startswith("multipart/"):
            parts = payload.get("parts", []) or []
            for p in parts:
                if p.get("mimeType") == "text/plain":
                    text = _extract_body(p)
                    if text:
                        return text
            for p in parts:
                text = _extract_body(p)
                if text:
                    return text
        if data:
            return _b64_decode(data)
        return ""
    except Exception:
        return ""


def _parse_msg(raw) -> dict:
    payload = raw.get("payload", {}) or {}
    headers_list = payload.get("headers", []) or []
    headers = {}
    for h in headers_list:
        name = h.get("name", "")
        value = h.get("value", "")
        if name:
            headers[name] = value
    headers_l = {k.lower(): v for k, v in headers.items()}

    date_str = headers_l.get("date", "")
    received_at = None
    if date_str:
        try:
            received_at = stdlib_email.utils.parsedate_to_datetime(date_str)
        except Exception:
            received_at = None
    if not received_at:
        received_at = datetime.utcnow()

    return {
        "gmail_message_id": raw.get("id"),
        "thread_id": raw.get("threadId"),
        "sender": headers_l.get("from", ""),
        "recipient": headers_l.get("to", ""),
        "subject": headers_l.get("subject", ""),
        "body": _extract_body(payload),
        "received_at": received_at,
        "raw_headers": headers,
    }


async def _gmail_get(token, path, params=None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GMAIL_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


async def _gmail_post(token, path, body) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GMAIL_BASE}{path}",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def _send_email(token, to, subject, body, thread_id=None) -> dict:
    subj = subject or ""
    if not subj.lower().startswith("re:"):
        subj = "Re: " + subj
    msg = MIMEText(body or "")
    msg["to"] = to
    msg["subject"] = subj

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    return await _gmail_post(token, "/users/me/messages/send", payload)


async def _fetch_unread(token, max_results=20) -> list:
    data = await _gmail_get(token, "/users/me/messages",
                            params={"q": "is:unread -from:me",
                                    "maxResults": max_results})
    msgs = data.get("messages", []) or []
    out = []
    for m in msgs:
        try:
            full = await _gmail_get(token, f"/users/me/messages/{m['id']}",
                                    params={"format": "full"})
            out.append(_parse_msg(full))
        except Exception:
            continue
    return out


async def _thread_context(token, thread_id, skip_id) -> str:
    try:
        data = await _gmail_get(token, f"/users/me/threads/{thread_id}")
        msgs = data.get("messages", []) or []
        ctx = []
        for m in msgs:
            if m.get("id") == skip_id:
                continue
            parsed = _parse_msg(m)
            body = (parsed.get("body") or "")[:300]
            ctx.append(f"FROM: {parsed.get('sender','')}\n{body}")
        if not ctx:
            return ""
        return "\n---\n".join(ctx[-3:])
    except Exception:
        return ""

# Section 8: Save Email to DB

async def save_email(db, user, raw, analysis) -> Optional[EmailRecord]:
    if raw.get("gmail_message_id"):
        res = await db.execute(select(EmailRecord).where(
            EmailRecord.gmail_message_id == raw["gmail_message_id"]
        ))
        if res.scalar_one_or_none():
            return None

    data = (analysis or {}).get("data", {})
    now = datetime.utcnow()
    follow_up_at = None
    if data.get("follow_up_needed") and data.get("follow_up_hours"):
        try:
            follow_up_at = now + timedelta(hours=float(data.get("follow_up_hours")))
        except Exception:
            follow_up_at = None

    record = EmailRecord(
        id=str(uuid.uuid4()),
        user_id=user.id,
        gmail_message_id=raw.get("gmail_message_id"),
        thread_id=raw.get("thread_id"),
        sender=raw.get("sender"),
        recipient=raw.get("recipient"),
        subject=raw.get("subject"),
        body=raw.get("body"),
        received_at=raw.get("received_at"),
        intent=data.get("intent"),
        priority=data.get("priority"),
        priority_score=data.get("priority_score"),
        sentiment=data.get("sentiment"),
        language=data.get("language"),
        summary=data.get("summary"),
        action_taken=data.get("action"),
        assigned_department=data.get("assigned_department"),
        confidence_score=data.get("confidence_score"),
        generated_reply=data.get("generated_reply"),
        reply_sent=False,
        reply_sent_at=None,
        follow_up_at=follow_up_at,
        escalated=bool(data.get("escalation_risk")),
        raw_headers=raw.get("raw_headers") or {},
        ai_metadata=data,
        processed_at=now,
        status="processed",
    )

    if (
        data.get("action") == "auto_reply"
        and (data.get("confidence_score") or 0) >= AUTO_THRESHOLD
        and user.access_token
        and raw.get("sender")
    ):
        try:
            await _send_email(
                user.access_token,
                raw.get("sender"),
                raw.get("subject"),
                data.get("generated_reply"),
                raw.get("thread_id"),
            )
            record.reply_sent = True
            record.reply_sent_at = now
            record.status = "auto_replied"
        except Exception as e:
            record.status = f"reply_failed: {str(e)[:100]}"

    db.add(record)
    db.add(ActivityLog(
        user_id=user.id,
        email_id=record.id,
        action="email_processed",
        details={"gmail_message_id": raw.get("gmail_message_id"),
                 "intent": data.get("intent"),
                 "priority": data.get("priority"),
                 "auto_reply": record.reply_sent},
    ))
    await db.commit()
    await db.refresh(record)
    return record

# Section 9: Pydantic Schemas

class EmailInput(BaseModel):
    sender: str
    sender_name: Optional[str] = ""
    subject: str
    body: str
    thread_context: Optional[str] = None


class BatchIn(BaseModel):
    emails: list[EmailInput]


class ReplyIn(BaseModel):
    body: str

class DevTokenIn(BaseModel):
    email: str

# Section 10: Routes

@app.get("/")
async def root():
    return {"message": "MailBrain API is running", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": f"error: {str(e)[:200]}"}


@app.get("/debug")
async def debug():
    routes = []
    for r in app.router.routes:
        methods = sorted(list(getattr(r, "methods", []) or []))
        path = getattr(r, "path", None) or str(getattr(r, "path_regex", ""))
        routes.append({"path": path, "methods": methods, "name": r.name})
    env = {
        "GOOGLE_CLIENT_ID": "SET" if os.environ.get("GOOGLE_CLIENT_ID") else "MISSING",
        "GOOGLE_CLIENT_SECRET": "SET" if os.environ.get("GOOGLE_CLIENT_SECRET") else "MISSING",
        "GOOGLE_REDIRECT_URI": "SET" if os.environ.get("GOOGLE_REDIRECT_URI") else "MISSING",
        "FRONTEND_URL": "SET" if os.environ.get("FRONTEND_URL") else "MISSING",
        "JWT_SECRET": "SET" if os.environ.get("JWT_SECRET") else "MISSING",
        "GEMINI_API_KEY": "SET" if os.environ.get("GEMINI_API_KEY") else "MISSING",
        "DATABASE_URL": "SET" if os.environ.get("DATABASE_URL") else "MISSING",
        "AI_MODEL": "SET" if os.environ.get("AI_MODEL") else "MISSING",
        "AUTO_SEND_CONFIDENCE_THRESHOLD": "SET" if os.environ.get("AUTO_SEND_CONFIDENCE_THRESHOLD") else "MISSING",
        "ALLOW_DEV_TOKEN": "SET" if os.environ.get("ALLOW_DEV_TOKEN") else "MISSING",
    }
    return {"routes": routes, "env": env}

# Auth Routes

@app.get("/auth/google")
async def auth_google():
    url = (
        f"{GOOGLE_AUTH_URL}"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES.replace(' ','%20')}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url=url, status_code=302)


@app.get("/auth/callback")
async def auth_callback(code: str = "", db: AsyncSession = Depends(get_db)):
    if not code:
        raise HTTPException(400, "Missing code")
    async with httpx.AsyncClient(timeout=30) as client:
        tok = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
        if tok.status_code != 200:
            return RedirectResponse(url="/auth/google", status_code=302)
        tokens = tok.json()
        prof = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        prof.raise_for_status()
        profile = prof.json()

    res = await db.execute(select(User).where(User.email == profile.get("email")))
    user = res.scalar_one_or_none()
    exp = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))

    if user:
        user.access_token = tokens.get("access_token")
        if tokens.get("refresh_token"):
            user.refresh_token = tokens.get("refresh_token")
        user.token_expiry = exp
        user.name = profile.get("name")
        user.picture = profile.get("picture")
    else:
        user = User(
            id=str(uuid.uuid4()),
            email=profile.get("email"),
            name=profile.get("name"),
            picture=profile.get("picture"),
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_expiry=exp,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)
    token = make_token(user.id, user.email)
    return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={token}", status_code=302)


@app.get("/auth/me")
async def auth_me(user: User = Depends(auth)):
    return {"id": user.id, "email": user.email, "name": user.name, "picture": user.picture}


@app.post("/auth/logout")
async def auth_logout():
    return {"message": "Delete JWT client-side"}

@app.post("/auth/dev-token")
async def auth_dev_token(data: DevTokenIn, db: AsyncSession = Depends(get_db)):
    if os.environ.get("ALLOW_DEV_TOKEN") != "1":
        raise HTTPException(403, "Dev token disabled")
    res = await db.execute(select(User).where(User.email == data.email))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    token = make_token(user.id, user.email)
    return {"token": token}


async def _refresh_google_token(db: AsyncSession, user: User) -> str:
    if not user.refresh_token:
        raise HTTPException(400, "No refresh token")
    async with httpx.AsyncClient(timeout=30) as client:
        tok = await client.post(GOOGLE_TOKEN_URL, data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": user.refresh_token,
            "grant_type": "refresh_token",
        })
    if tok.status_code != 200:
        raise HTTPException(400, f"Refresh failed: {tok.text[:200]}")
    data = tok.json()
    user.access_token = data.get("access_token")
    user.token_expiry = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))
    await db.commit()
    return user.access_token


@app.post("/auth/refresh")
async def auth_refresh(user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    token = await _refresh_google_token(db, user)
    return {"access_token": token, "token_expiry": user.token_expiry.isoformat()}

# Email Routes

@app.get("/emails/")
async def list_emails(
    page: int = 1,
    page_size: int = 20,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    user: User = Depends(auth),
    db: AsyncSession = Depends(get_db),
):
    filters = [EmailRecord.user_id == user.id]
    if intent:
        filters.append(EmailRecord.intent == intent)
    if priority:
        filters.append(EmailRecord.priority == priority)

    total = (await db.execute(select(func.count(EmailRecord.id)).where(*filters))).scalar() or 0
    rows = (
        await db.execute(
            select(EmailRecord)
            .where(*filters)
            .order_by(desc(EmailRecord.processed_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

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
                "action_taken": e.action_taken,
                "assigned_department": e.assigned_department,
                "confidence_score": e.confidence_score,
                "reply_sent": e.reply_sent,
                "escalated": e.escalated,
                "received_at": str(e.received_at),
                "processed_at": str(e.processed_at),
                "status": e.status,
            }
            for e in rows
        ],
    }


@app.post("/emails/sync")
async def sync_emails(
    max_results: int = 20,
    user: User = Depends(auth),
    db: AsyncSession = Depends(get_db),
):
    if not user.access_token:
        raise HTTPException(400, "No Gmail token. Visit /auth/google to reconnect.")

    try:
        raw_emails = await _fetch_unread(user.access_token, max_results)
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 401:
            if user.refresh_token:
                await _refresh_google_token(db, user)
                raw_emails = await _fetch_unread(user.access_token, max_results)
            else:
                raise HTTPException(401, "Gmail token expired. Reconnect.")
        else:
            raise HTTPException(400, f"Gmail fetch failed: {str(e)[:200]}")
    except Exception as e:
        raise HTTPException(400, f"Gmail fetch failed: {str(e)[:200]}")

    if not raw_emails:
        return {"processed": 0, "skipped": 0, "errors": 0, "emails": [], "error_details": []}

    done, skipped, errors = [], [], []
    for raw in raw_emails:
        try:
            ctx = None
            if raw.get("thread_id") and raw.get("gmail_message_id"):
                ctx = await _thread_context(user.access_token, raw["thread_id"], raw["gmail_message_id"]) or None
            analysis = await ai_analyze(raw.get("subject", ""), raw.get("body", ""), raw.get("sender", ""), ctx)
            saved = await save_email(db, user, raw, analysis)
            if saved is None:
                skipped.append(raw.get("subject", "?"))
                continue
            done.append({
                "id": saved.id,
                "subject": saved.subject,
                "sender": saved.sender,
                "intent": saved.intent,
                "priority": saved.priority,
                "action": saved.action_taken,
                "reply_sent": saved.reply_sent,
            })
            if raw.get("gmail_message_id"):
                try:
                    await _gmail_post(
                        user.access_token,
                        f"/users/me/messages/{raw['gmail_message_id']}/modify",
                        {"removeLabelIds": ["UNREAD"]},
                    )
                except Exception:
                    pass
        except Exception as e:
            errors.append({"subject": raw.get("subject", "?"), "error": str(e)})

    return {
        "processed": len(done),
        "skipped": len(skipped),
        "errors": len(errors),
        "emails": done,
        "error_details": errors,
    }


@app.post("/emails/process")
async def process_email(
    data: EmailInput,
    user: User = Depends(auth),
    db: AsyncSession = Depends(get_db),
):
    analysis = await ai_analyze(data.subject, data.body, data.sender, data.thread_context)
    raw = {
        "sender": data.sender,
        "subject": data.subject,
        "body": data.body,
        "received_at": datetime.utcnow(),
        "raw_headers": {},
    }
    saved = await save_email(db, user, raw, analysis)
    return {
        "id": saved.id,
        "analysis": analysis["data"],
        "reply_sent": saved.reply_sent,
        "status": saved.status,
    }


@app.post("/emails/batch")
async def batch_emails(
    batch: BatchIn,
    user: User = Depends(auth),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for e in batch.emails:
        analysis = await ai_analyze(e.subject, e.body, e.sender, e.thread_context)
        raw = {
            "sender": e.sender,
            "subject": e.subject,
            "body": e.body,
            "received_at": datetime.utcnow(),
            "raw_headers": {},
        }
        saved = await save_email(db, user, raw, analysis)
        if saved:
            results.append({"id": saved.id, "intent": saved.intent, "priority": saved.priority})
    return {"processed": len(results), "emails": results}


@app.get("/emails/{email_id}")
async def get_email(email_id: str, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id,
        EmailRecord.user_id == user.id,
    ))
    e = res.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    return {
        "id": e.id,
        "sender": e.sender,
        "recipient": e.recipient,
        "subject": e.subject,
        "body": e.body,
        "received_at": str(e.received_at),
        "intent": e.intent,
        "priority": e.priority,
        "priority_score": e.priority_score,
        "sentiment": e.sentiment,
        "language": e.language,
        "summary": e.summary,
        "action_taken": e.action_taken,
        "assigned_department": e.assigned_department,
        "confidence_score": e.confidence_score,
        "generated_reply": e.generated_reply,
        "reply_sent": e.reply_sent,
        "reply_sent_at": str(e.reply_sent_at),
        "escalated": e.escalated,
        "follow_up_at": str(e.follow_up_at),
        "raw_headers": e.raw_headers,
        "ai_metadata": e.ai_metadata,
        "status": e.status,
    }


@app.post("/emails/{email_id}/approve")
async def approve_reply(email_id: str, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id,
        EmailRecord.user_id == user.id,
    ))
    e = res.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    if e.reply_sent:
        raise HTTPException(400, "Reply already sent")
    if not e.generated_reply:
        raise HTTPException(400, "No reply to send")
    if not user.access_token:
        raise HTTPException(400, "No Gmail token")

    await _send_email(user.access_token, e.sender, e.subject, e.generated_reply, e.thread_id)
    e.reply_sent = True
    e.reply_sent_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}


@app.post("/emails/{email_id}/reply")
async def send_reply(
    email_id: str,
    reply: ReplyIn,
    user: User = Depends(auth),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id,
        EmailRecord.user_id == user.id,
    ))
    e = res.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    if not user.access_token:
        raise HTTPException(400, "No Gmail token")

    await _send_email(user.access_token, e.sender, e.subject, reply.body, e.thread_id)
    e.reply_sent = True
    e.reply_sent_at = datetime.utcnow()
    e.generated_reply = reply.body
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}

# Analytics Routes

@app.get("/analytics/overview")
async def analytics_overview(days: int = 7, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    w = [EmailRecord.user_id == user.id, EmailRecord.processed_at >= since]
    total = (await db.execute(select(func.count(EmailRecord.id)).where(*w))).scalar() or 0
    critical = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.priority == "CRITICAL"))).scalar() or 0
    auto_r = (
        await db.execute(
            select(func.count(EmailRecord.id)).where(*w, EmailRecord.action_taken == "auto_reply", EmailRecord.reply_sent == True)
        )
    ).scalar() or 0
    escalated = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.escalated == True))).scalar() or 0
    avg_c = (await db.execute(select(func.avg(EmailRecord.confidence_score)).where(*w))).scalar() or 0

    return {
        "period_days": days,
        "total_emails": total,
        "critical_emails": critical,
        "auto_replied": auto_r,
        "escalated": escalated,
        "automation_rate": round(auto_r / total * 100 if total else 0, 1),
        "avg_confidence": round(float(avg_c), 3),
    }


@app.get("/analytics/intent")
async def analytics_intent(days: int = 30, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        await db.execute(
            select(EmailRecord.intent, func.count(EmailRecord.id).label("count"))
            .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
            .group_by(EmailRecord.intent)
            .order_by(desc("count"))
        )
    ).all()
    return {"distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]}


@app.get("/analytics/priority")
async def analytics_priority(days: int = 7, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        await db.execute(
            select(EmailRecord.priority, func.count(EmailRecord.id).label("count"))
            .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
            .group_by(EmailRecord.priority)
        )
    ).all()
    order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
    breakdown = [{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows]
    breakdown.sort(key=lambda x: order.get(x["priority"], 9))
    return {"breakdown": breakdown}


@app.get("/analytics/trends")
async def analytics_trends(days: int = 14, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        await db.execute(
            select(func.date(EmailRecord.processed_at).label("date"), func.count(EmailRecord.id).label("total"))
            .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
            .group_by(func.date(EmailRecord.processed_at))
            .order_by(func.date(EmailRecord.processed_at))
        )
    ).all()
    return {"trends": [{"date": str(r.date), "total": r.total} for r in rows]}


@app.get("/analytics/automation")
async def analytics_automation(user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    rows = (
        await db.execute(
            select(EmailRecord.action_taken, func.count(EmailRecord.id).label("count"))
            .where(EmailRecord.user_id == user.id)
            .group_by(EmailRecord.action_taken)
            .order_by(desc("count"))
        )
    ).all()
    total = sum(r.count for r in rows)
    return {
        "total_processed": total,
        "actions": [
            {
                "action": r.action_taken or "none",
                "count": r.count,
                "percentage": round(r.count / total * 100, 1) if total else 0,
            }
            for r in rows
        ],
    }


@app.get("/analytics/escalations")
async def analytics_escalations(days: int = 7, user: User = Depends(auth), db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        await db.execute(
            select(EmailRecord)
            .where(
                EmailRecord.user_id == user.id,
                EmailRecord.processed_at >= since,
                (EmailRecord.escalated == True) | (EmailRecord.priority == "CRITICAL"),
            )
            .order_by(desc(EmailRecord.priority_score))
        )
    ).scalars().all()
    return {
        "count": len(rows),
        "emails": [
            {
                "id": e.id,
                "sender": e.sender,
                "subject": e.subject,
                "summary": e.summary,
                "priority": e.priority,
                "intent": e.intent,
                "escalated": e.escalated,
                "reply_sent": e.reply_sent,
                "processed_at": str(e.processed_at),
            }
            for e in rows
        ],
    }

# Webhook Route

@app.post("/webhooks/gmail")
async def webhook_gmail(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        body = await request.json()
        msg = body.get("message", {})
        raw = base64.b64decode(msg.get("data", "") + "==").decode()
        payload = json.loads(raw)
    except Exception:
        return {"status": "ignored"}

    email_addr = payload.get("emailAddress")
    if not email_addr:
        return {"status": "ignored"}

    res = await db.execute(select(User).where(User.email == email_addr))
    user = res.scalar_one_or_none()
    if not user or not user.access_token:
        return {"status": "ignored"}

    try:
        raw_emails = await _fetch_unread(user.access_token, 5)
        for raw_email in raw_emails:
            ctx = None
            if raw_email.get("thread_id") and raw_email.get("gmail_message_id"):
                ctx = await _thread_context(user.access_token, raw_email["thread_id"], raw_email["gmail_message_id"]) or None
            analysis = await ai_analyze(
                raw_email.get("subject", ""),
                raw_email.get("body", ""),
                raw_email.get("sender", ""),
                ctx,
            )
            saved = await save_email(db, user, raw_email, analysis)
            if saved and raw_email.get("gmail_message_id"):
                try:
                    await _gmail_post(
                        user.access_token,
                        f"/users/me/messages/{raw_email['gmail_message_id']}/modify",
                        {"removeLabelIds": ["UNREAD"]},
                    )
                except Exception:
                    pass
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "processed"}
