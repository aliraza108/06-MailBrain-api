import sys
import os

# Make sure current directory is in path so Vercel can find all modules
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import traceback


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import init_db
        await init_db()
        print("✅ DB initialized")
    except Exception as e:
        print(f"❌ DB init failed: {e}")
        traceback.print_exc()
    yield


app = FastAPI(
    title="MailBrain API",
    description="Autonomous Email Operations Manager",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://06-mailbrain.vercel.app",
        "http://06-mailbrain.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import routers using flat imports (not package-style) ──────────────────────
try:
    from routes.auth import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    print("✅ auth router loaded")
except Exception as e:
    print(f"❌ auth router failed: {e}")
    traceback.print_exc()

try:
    from routes.emails import router as emails_router
    app.include_router(emails_router, prefix="/emails", tags=["Emails"])
    print("✅ emails router loaded")
except Exception as e:
    print(f"❌ emails router failed: {e}")
    traceback.print_exc()

try:
    from routes.analytics import router as analytics_router
    app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
    print("✅ analytics router loaded")
except Exception as e:
    print(f"❌ analytics router failed: {e}")
    traceback.print_exc()

try:
    from routes.webhooks import router as webhooks_router
    app.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
    print("✅ webhooks router loaded")
except Exception as e:
    print(f"❌ webhooks router failed: {e}")
    traceback.print_exc()


@app.get("/")
async def root():
    return {"message": "MailBrain API is running", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    try:
        from sqlalchemy import text
        from database import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": str(e)}