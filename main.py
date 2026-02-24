"""
MailBrain API - Complete single file deployment for Vercel
"""
import os, sys, uuid, base64, json, re, httpx, jwt, traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional
from email.mime.text import MIMEText
import email as email_lib

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON, func, desc, select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "https://06-mailbrain-api.vercel.app/auth/callback")
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "https://06-mailbrain.vercel.app")
JWT_SECRET           = os.environ.get("JWT_SECRET", "fallback-secret-change-me")
JWT_ALGORITHM        = "HS256"
JWT_EXPIRE_HOURS     = 24
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY", "")
AI_MODEL             = os.environ.get("AI_MODEL", "gemini-2.5-flash")
AI_BASE_URL          = "https://generativelanguage.googleapis.com/v1beta/openai/"
AUTO_SEND_THRESHOLD  = float(os.environ.get("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))
DATABASE_URL         = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./mailbrain.db")

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_API        = "https://gmail.googleapis.com/gmail/v1"

GMAIL_SCOPES = " ".join([
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
])

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_connect_args = {"ssl": "require"} if "postgresql" in DATABASE_URL else {}
engine = create_async_engine(DATABASE_URL, echo=False, connect_args=_connect_args)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
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
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String)
    email_id   = Column(String)
    action     = Column(String)
    details    = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


async def get_db():
    async with SessionLocal() as session:
        yield session


# ═══════════════════════════════════════════════════════════════════════════════
# APP STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables ready")
    except Exception as e:
        print(f"❌ DB startup error: {e}")
        traceback.print_exc()
    yield


app = FastAPI(
    title="MailBrain API",
    description="Autonomous Email Operations Manager",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_jwt(user_id: str, email: str) -> str:
    return jwt.encode(
        {"sub": user_id, "email": email,
         "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
         "iat": datetime.utcnow()},
        JWT_SECRET, algorithm=JWT_ALGORITHM
    )


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


async def current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required")
    payload = decode_jwt(authorization[7:])
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ═══════════════════════════════════════════════════════════════════════════════
# GMAIL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_body(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            result = _extract_body(part)
            if result:
                return result
    return ""


def _parse_gmail(raw: dict) -> dict:
    headers = {h["name"].lower(): h["value"]
               for h in raw.get("payload", {}).get("headers", [])}
    try:
        received_at = email_lib.utils.parsedate_to_datetime(headers.get("date", ""))
    except Exception:
        received_at = datetime.utcnow()
    return {
        "gmail_message_id": raw.get("id"),
        "thread_id":        raw.get("threadId"),
        "sender":           headers.get("from", ""),
        "recipient":        headers.get("to", ""),
        "subject":          headers.get("subject", "(no subject)"),
        "body":             _extract_body(raw.get("payload", {})),
        "received_at":      received_at,
        "raw_headers":      headers,
    }


async def _gmail_get(token: str, path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{GMAIL_API}{path}",
                        headers={"Authorization": f"Bearer {token}"},
                        params=params or {})
        r.raise_for_status()
        return r.json()


async def _gmail_send(token: str, to: str, subject: str, body: str, thread_id: str = None):
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{GMAIL_API}/users/me/messages/send",
                         headers={"Authorization": f"Bearer {token}",
                                  "Content-Type": "application/json"},
                         json=payload)
        r.raise_for_status()
        return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

async def ai_analyze(subject: str, body: str, sender: str, thread_ctx: str = None) -> dict:
    prompt = f"""Analyze this email. Reply ONLY with valid JSON, no markdown, no backticks.

FROM: {sender}
SUBJECT: {subject}
BODY: {body[:3000]}
{"THREAD: " + thread_ctx[:800] if thread_ctx else ""}

JSON format:
{{
  "intent": "support_request|refund_demand|sales_inquiry|meeting_request|complaint|spam|urgent_escalation|billing_question|partnership_offer|general_inquiry",
  "priority": "CRITICAL|HIGH|NORMAL|LOW",
  "priority_score": 0.8,
  "sentiment": "positive|neutral|negative",
  "language": "en",
  "summary": "one sentence summary",
  "action": "auto_reply|assign_department|create_ticket|schedule_meeting|flag_management|request_info",
  "assigned_department": "support|billing|sales|management|technical",
  "confidence_score": 0.85,
  "escalation_risk": false,
  "follow_up_needed": false,
  "follow_up_hours": null,
  "reply_tone": "professional|empathetic|friendly|firm",
  "generated_reply": "Complete ready-to-send reply text here",
  "keywords_detected": []
}}"""

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{AI_BASE_URL}chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": AI_MODEL, "max_tokens": 2000,
                      "messages": [{"role": "user", "content": prompt}]},
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```json\s*|^```\s*|\s*```$", "", content).strip()
            return {"success": True, "data": json.loads(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _save_email(db: AsyncSession, user: User, raw: dict, analysis: dict) -> EmailRecord:
    d = analysis.get("data", {})
    follow_up_at = None
    if d.get("follow_up_needed") and d.get("follow_up_hours"):
        follow_up_at = datetime.utcnow() + timedelta(hours=float(d["follow_up_hours"]))

    rec = EmailRecord(
        id=str(uuid.uuid4()), user_id=user.id,
        gmail_message_id=raw.get("gmail_message_id"),
        thread_id=raw.get("thread_id"),
        sender=raw.get("sender", ""), recipient=raw.get("recipient", ""),
        subject=raw.get("subject", ""), body=raw.get("body", ""),
        received_at=raw.get("received_at"),
        intent=d.get("intent"), priority=d.get("priority"),
        priority_score=d.get("priority_score"), sentiment=d.get("sentiment"),
        language=d.get("language", "en"), summary=d.get("summary"),
        action_taken=d.get("action"), assigned_department=d.get("assigned_department"),
        confidence_score=d.get("confidence_score"), generated_reply=d.get("generated_reply"),
        escalated=bool(d.get("escalation_risk", False)), follow_up_at=follow_up_at,
        ai_metadata={"keywords": d.get("keywords_detected", []), "tone": d.get("reply_tone")},
        raw_headers=raw.get("raw_headers", {}), status="processed",
    )

    if (d.get("action") == "auto_reply"
            and float(d.get("confidence_score") or 0) >= AUTO_SEND_THRESHOLD
            and user.access_token and raw.get("sender")):
        try:
            await _gmail_send(user.access_token, raw["sender"],
                              raw.get("subject", ""), d.get("generated_reply", ""),
                              raw.get("thread_id"))
            rec.reply_sent = True
            rec.reply_sent_at = datetime.utcnow()
        except Exception as e:
            rec.status = f"reply_failed:{str(e)[:80]}"

    db.add(rec)
    db.add(ActivityLog(user_id=user.id, email_id=rec.id,
                       action=d.get("action", "processed"),
                       details={"intent": d.get("intent"),
                                "priority": d.get("priority"),
                                "auto_sent": rec.reply_sent}))
    await db.commit()
    await db.refresh(rec)
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — BASE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "MailBrain API is running", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/auth/google", tags=["Auth"])
async def auth_google():
    """Redirect to Google OAuth consent screen"""
    url = (
        f"{GOOGLE_AUTH_URL}?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={GMAIL_SCOPES.replace(' ', '%20')}"
        f"&access_type=offline&prompt=consent"
    )
    return RedirectResponse(url=url, status_code=302)


@app.get("/auth/callback", tags=["Auth"])
async def auth_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Google OAuth callback — exchanges code for tokens, creates user, returns JWT"""
    async with httpx.AsyncClient(timeout=30) as c:
        # Exchange code for tokens
        tok_r = await c.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
        if tok_r.status_code != 200:
            raise HTTPException(400, f"Token exchange failed: {tok_r.text}")
        tokens = tok_r.json()

        # Get user profile
        usr_r = await c.get(GOOGLE_USER_URL,
                             headers={"Authorization": f"Bearer {tokens['access_token']}"})
        info = usr_r.json()

    # Upsert user in DB
    res = await db.execute(select(User).where(User.email == info["email"]))
    user = res.scalar_one_or_none()
    expiry = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))

    if user:
        user.access_token  = tokens.get("access_token")
        user.refresh_token = tokens.get("refresh_token", user.refresh_token)
        user.token_expiry  = expiry
        user.name          = info.get("name")
        user.picture       = info.get("picture")
    else:
        user = User(id=str(uuid.uuid4()), email=info["email"],
                    name=info.get("name"), picture=info.get("picture"),
                    access_token=tokens.get("access_token"),
                    refresh_token=tokens.get("refresh_token"),
                    token_expiry=expiry)
        db.add(user)

    await db.commit()
    await db.refresh(user)

    token = make_jwt(user.id, user.email)
    return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={token}", status_code=302)


@app.get("/auth/me", tags=["Auth"])
async def auth_me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "picture": user.picture}


@app.post("/auth/logout", tags=["Auth"])
async def auth_logout():
    return {"message": "Logged out — delete your token on the client"}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — EMAILS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/emails/", tags=["Emails"])
async def list_emails(
    page: int = 1, page_size: int = 20,
    intent: Optional[str] = None, priority: Optional[str] = None,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    q = select(EmailRecord).where(EmailRecord.user_id == user.id).order_by(desc(EmailRecord.processed_at))
    if intent:   q = q.where(EmailRecord.intent == intent)
    if priority: q = q.where(EmailRecord.priority == priority)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows  = (await db.execute(q.offset((page - 1) * page_size).limit(page_size))).scalars().all()

    return {
        "total": total, "page": page, "page_size": page_size,
        "emails": [{
            "id": e.id, "sender": e.sender, "subject": e.subject,
            "summary": e.summary, "intent": e.intent, "priority": e.priority,
            "priority_score": e.priority_score, "sentiment": e.sentiment,
            "action_taken": e.action_taken, "confidence_score": e.confidence_score,
            "reply_sent": e.reply_sent, "escalated": e.escalated,
            "received_at": str(e.received_at), "processed_at": str(e.processed_at),
            "status": e.status,
        } for e in rows],
    }


@app.post("/emails/sync", tags=["Emails"])
async def sync_emails(
    max_results: int = 20,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    if not user.access_token:
        raise HTTPException(400, "No Gmail token — reconnect via /auth/google")
    try:
        data = await _gmail_get(user.access_token, "/users/me/messages",
                                {"maxResults": max_results, "q": "is:unread -from:me"})
    except Exception as e:
        raise HTTPException(400, f"Gmail error: {e}")

    messages = data.get("messages", [])
    if not messages:
        return {"message": "No new emails", "processed": 0}

    done, errs = [], []
    for ref in messages:
        try:
            raw_msg  = await _gmail_get(user.access_token, f"/users/me/messages/{ref['id']}", {"format": "full"})
            parsed   = _parse_gmail(raw_msg)
            analysis = await ai_analyze(parsed["subject"], parsed["body"], parsed["sender"])
            if analysis["success"]:
                saved = await _save_email(db, user, parsed, analysis)
                done.append({"id": saved.id, "subject": saved.subject,
                             "intent": saved.intent, "priority": saved.priority})
                async with httpx.AsyncClient(timeout=10) as c:
                    await c.post(f"{GMAIL_API}/users/me/messages/{ref['id']}/modify",
                                 headers={"Authorization": f"Bearer {user.access_token}"},
                                 json={"removeLabelIds": ["UNREAD"]})
            else:
                errs.append(analysis.get("error"))
        except Exception as e:
            errs.append(str(e))

    return {"processed": len(done), "errors": len(errs), "emails": done}


@app.post("/emails/process", tags=["Emails"])
async def process_email(
    data: EmailInput,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    analysis = await ai_analyze(data.subject, data.body, data.sender, data.thread_context)
    if not analysis["success"]:
        raise HTTPException(500, f"AI error: {analysis.get('error')}")
    raw = {"sender": data.sender, "subject": data.subject,
           "body": data.body, "received_at": datetime.utcnow(), "raw_headers": {}}
    saved = await _save_email(db, user, raw, analysis)
    return {"id": saved.id, "analysis": analysis["data"], "reply_sent": saved.reply_sent}


@app.post("/emails/batch", tags=["Emails"])
async def batch_process(
    batch: BatchIn,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    results = []
    for e in batch.emails:
        analysis = await ai_analyze(e.subject, e.body, e.sender, e.thread_context)
        if analysis["success"]:
            raw = {"sender": e.sender, "subject": e.subject,
                   "body": e.body, "received_at": datetime.utcnow(), "raw_headers": {}}
            saved = await _save_email(db, user, raw, analysis)
            results.append({"id": saved.id, "intent": saved.intent, "priority": saved.priority})
    return {"processed": len(results), "emails": results}


@app.get("/emails/{email_id}", tags=["Emails"])
async def get_email(
    email_id: str,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id, EmailRecord.user_id == user.id))
    e = res.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    return {
        "id": e.id, "sender": e.sender, "recipient": e.recipient,
        "subject": e.subject, "body": e.body, "received_at": str(e.received_at),
        "intent": e.intent, "priority": e.priority, "priority_score": e.priority_score,
        "sentiment": e.sentiment, "language": e.language, "summary": e.summary,
        "action_taken": e.action_taken, "assigned_department": e.assigned_department,
        "confidence_score": e.confidence_score, "generated_reply": e.generated_reply,
        "reply_sent": e.reply_sent, "reply_sent_at": str(e.reply_sent_at),
        "escalated": e.escalated, "ai_metadata": e.ai_metadata, "status": e.status,
    }


@app.post("/emails/{email_id}/approve", tags=["Emails"])
async def approve_reply(
    email_id: str,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id, EmailRecord.user_id == user.id))
    e = res.scalar_one_or_none()
    if not e:             raise HTTPException(404, "Email not found")
    if e.reply_sent:      raise HTTPException(400, "Reply already sent")
    if not e.generated_reply: raise HTTPException(400, "No generated reply to send")

    await _gmail_send(user.access_token, e.sender, e.subject, e.generated_reply, e.thread_id)
    e.reply_sent = True
    e.reply_sent_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}


@app.post("/emails/{email_id}/reply", tags=["Emails"])
async def send_reply(
    email_id: str, reply: ReplyIn,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(EmailRecord).where(
        EmailRecord.id == email_id, EmailRecord.user_id == user.id))
    e = res.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Email not found")
    await _gmail_send(user.access_token, e.sender, e.subject, reply.body, e.thread_id)
    e.reply_sent = True
    e.reply_sent_at = datetime.utcnow()
    e.generated_reply = reply.body
    await db.commit()
    return {"message": "Reply sent", "email_id": email_id}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/analytics/overview", tags=["Analytics"])
async def analytics_overview(
    days: int = 7,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    w = [EmailRecord.user_id == user.id, EmailRecord.processed_at >= since]

    total    = (await db.execute(select(func.count(EmailRecord.id)).where(*w))).scalar() or 0
    critical = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.priority == "CRITICAL"))).scalar() or 0
    auto_r   = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.reply_sent == True))).scalar() or 0
    esc      = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.escalated == True))).scalar() or 0
    avg_c    = (await db.execute(select(func.avg(EmailRecord.confidence_score)).where(*w))).scalar() or 0

    return {
        "period_days": days, "total_emails": total, "critical_emails": critical,
        "auto_replied": auto_r, "escalated": esc,
        "automation_rate": round(auto_r / total * 100 if total else 0, 1),
        "avg_confidence": round(float(avg_c) * 100, 1),
    }


@app.get("/analytics/intent", tags=["Analytics"])
async def analytics_intent(
    days: int = 30,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord.intent, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(EmailRecord.intent).order_by(desc("count"))
    )).all()
    return {"distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]}


@app.get("/analytics/priority", tags=["Analytics"])
async def analytics_priority(
    days: int = 7,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord.priority, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(EmailRecord.priority)
    )).all()
    order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
    return {"breakdown": sorted(
        [{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows],
        key=lambda x: order.get(x["priority"], 9)
    )}


@app.get("/analytics/trends", tags=["Analytics"])
async def analytics_trends(
    days: int = 14,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(func.date(EmailRecord.processed_at).label("date"),
               func.count(EmailRecord.id).label("total"))
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(func.date(EmailRecord.processed_at))
        .order_by(func.date(EmailRecord.processed_at))
    )).all()
    return {"trends": [{"date": str(r.date), "total": r.total} for r in rows]}


@app.get("/analytics/automation", tags=["Analytics"])
async def analytics_automation(
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(EmailRecord.action_taken, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id)
        .group_by(EmailRecord.action_taken).order_by(desc("count"))
    )).all()
    total = sum(r.count for r in rows)
    return {"total_processed": total, "actions": [
        {"action": r.action_taken or "none", "count": r.count,
         "percentage": round(r.count / total * 100, 1) if total else 0}
        for r in rows
    ]}


@app.get("/analytics/escalations", tags=["Analytics"])
async def analytics_escalations(
    days: int = 7,
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord)
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since,
               (EmailRecord.escalated == True) | (EmailRecord.priority == "CRITICAL"))
        .order_by(desc(EmailRecord.priority_score))
    )).scalars().all()
    return {"count": len(rows), "emails": [
        {"id": e.id, "sender": e.sender, "subject": e.subject,
         "priority": e.priority, "escalated": e.escalated,
         "reply_sent": e.reply_sent, "processed_at": str(e.processed_at)}
        for e in rows
    ]}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhooks/gmail", tags=["Webhooks"])
async def webhook_gmail(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        body    = await request.json()
        msg     = body.get("message", {})
        payload = json.loads(base64.b64decode(msg.get("data", "") + "==").decode())
    except Exception:
        return {"status": "ignored"}

    email_addr = payload.get("emailAddress")
    if not email_addr:
        return {"status": "ignored"}

    res = await db.execute(select(User).where(User.email == email_addr))
    user = res.scalar_one_or_none()
    if not user or not user.access_token:
        return {"status": "user_not_found"}

    try:
        data = await _gmail_get(user.access_token, "/users/me/messages",
                                {"maxResults": 5, "q": "is:unread -from:me"})
        for ref in data.get("messages", []):
            raw_msg  = await _gmail_get(user.access_token, f"/users/me/messages/{ref['id']}", {"format": "full"})
            parsed   = _parse_gmail(raw_msg)
            analysis = await ai_analyze(parsed["subject"], parsed["body"], parsed["sender"])
            if analysis["success"]:
                await _save_email(db, user, parsed, analysis)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "processed"}