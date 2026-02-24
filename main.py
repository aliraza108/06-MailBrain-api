"""
MailBrain API — main.py
Entry point for Vercel Python serverless deployment.
"""
import sys
import os
import traceback
from contextlib import asynccontextmanager

# ── Ensure project root is on sys.path so all modules resolve ─────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ── Lifespan: init DB on cold start ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import init_db
        await init_db()
        print("✅ MailBrain DB ready")
    except Exception:
        print("❌ DB init failed:")
        traceback.print_exc()
    yield
    print("MailBrain shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MailBrain API",
    description="Autonomous Email Operations Manager — AI-powered inbox automation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler — returns JSON not HTML on crashes ───────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"❌ Unhandled error on {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url)},
    )


# ── Register routers ──────────────────────────────────────────────────────────

def _register_routers():
    errors = []

    try:
        from routes.auth import router as auth_router
        app.include_router(auth_router, prefix="/auth", tags=["Auth"])
        print("✅ auth router registered")
    except Exception as e:
        errors.append(f"auth: {e}")
        traceback.print_exc()

    try:
        from routes.emails import router as emails_router
        app.include_router(emails_router, prefix="/emails", tags=["Emails"])
        print("✅ emails router registered")
    except Exception as e:
        errors.append(f"emails: {e}")
        traceback.print_exc()

    try:
        from routes.analytics import router as analytics_router
        app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
        print("✅ analytics router registered")
    except Exception as e:
        errors.append(f"analytics: {e}")
        traceback.print_exc()

    try:
        from routes.webhooks import router as webhooks_router
        app.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
        print("✅ webhooks router registered")
    except Exception as e:
        errors.append(f"webhooks: {e}")
        traceback.print_exc()

    if errors:
        print(f"⚠️  Router errors: {errors}")


_register_routers()


# ── Base routes ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Base"])
async def root():
    return {
        "message": "MailBrain API is running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Base"])
async def health():
    try:
        from sqlalchemy import text
        from database import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status":   "healthy" if "error" not in db_status else "degraded",
        "database": db_status,
        "version":  "1.0.0",
    }


@app.get("/debug", tags=["Base"])
async def debug():
    """Shows loaded routes and env var presence — remove in production."""
    routes = [{"path": r.path, "methods": list(r.methods)} for r in app.routes]
    env_check = {
        "GOOGLE_CLIENT_ID":     bool(os.environ.get("GOOGLE_CLIENT_ID")),
        "GOOGLE_CLIENT_SECRET": bool(os.environ.get("GOOGLE_CLIENT_SECRET")),
        "GOOGLE_REDIRECT_URI":  os.environ.get("GOOGLE_REDIRECT_URI", "NOT SET"),
        "FRONTEND_URL":         os.environ.get("FRONTEND_URL", "NOT SET"),
        "JWT_SECRET":           bool(os.environ.get("JWT_SECRET")),
        "GEMINI_API_KEY":       bool(os.environ.get("GEMINI_API_KEY")),
        "DATABASE_URL":         os.environ.get("DATABASE_URL", "NOT SET")[:40] + "...",
    }
    return {"routes": routes, "env": env_check}