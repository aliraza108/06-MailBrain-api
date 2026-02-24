import os
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./mailbrain.db")

# Build SSL connect args — only for real PostgreSQL connections
_is_postgres = DATABASE_URL.startswith("postgresql")
_connect_args = {"ssl": "require"} if _is_postgres else {}

# Replace postgres:// with postgresql+asyncpg:// if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables. Safe to call multiple times."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()