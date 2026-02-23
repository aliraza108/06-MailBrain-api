from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./mailbrain.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

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


class Email(Base):
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

    # AI Analysis
    intent              = Column(String)        # support_request | refund | sales | meeting | complaint | spam | urgent
    priority            = Column(String)        # CRITICAL | HIGH | NORMAL | LOW
    priority_score      = Column(Float)
    sentiment           = Column(String)        # positive | neutral | negative
    language            = Column(String, default="en")
    summary             = Column(Text)
    action_taken        = Column(String)        # auto_reply | assigned | ticket | meeting | flagged | info_request
    assigned_department = Column(String)
    confidence_score    = Column(Float)

    # Response
    generated_reply     = Column(Text)
    reply_sent          = Column(Boolean, default=False)
    reply_sent_at       = Column(DateTime)

    # Workflow
    ticket_id           = Column(String)
    meeting_scheduled   = Column(Boolean, default=False)
    follow_up_at        = Column(DateTime)
    escalated           = Column(Boolean, default=False)

    # Metadata
    raw_headers         = Column(JSON)
    ai_metadata         = Column(JSON)
    processed_at        = Column(DateTime, default=datetime.utcnow)
    status              = Column(String, default="processed")  # processed | pending | error


class AutomationRule(Base):
    __tablename__ = "automation_rules"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String)
    name        = Column(String)
    condition   = Column(JSON)   # {"intent": "refund", "priority": "HIGH"}
    action      = Column(JSON)   # {"type": "assign", "department": "billing"}
    approved    = Column(Integer, default=0)   # learned approval count
    created_at  = Column(DateTime, default=datetime.utcnow)


class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String)
    email_id    = Column(String)
    action      = Column(String)
    details     = Column(JSON)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
