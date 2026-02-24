"""
MailBrain API — main.py
"""
import os
import sys
import traceback
from contextlib import asynccontextmanager

# Patch path FIRST
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import init_db
        await init_db()
        print("[MailBrain] DB ready")
    except Exception:
        traceback.print_exc()
    yield


app = FastAPI(
    title="MailBrain API",
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


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})


# Direct imports — no importlib
_errors = []

try:
    from routes.auth import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    print("[MailBrain] auth OK")
except Exception as e:
    _errors.append(f"auth: {e}")
    traceback.print_exc()

try:
    from routes.emails import router as emails_router
    app.include_router(emails_router, prefix="/emails", tags=["Emails"])
    print("[MailBrain] emails OK")
except Exception as e:
    _errors.append(f"emails: {e}")
    traceback.print_exc()

try:
    from routes.analytics import router as analytics_router
    app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
    print("[MailBrain] analytics OK")
except Exception as e:
    _errors.append(f"analytics: {e}")
    traceback.print_exc()

try:
    from routes.webhooks import router as webhooks_router
    app.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
    print("[MailBrain] webhooks OK")
except Exception as e:
    _errors.append(f"webhooks: {e}")
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
        db = "connected"
    except Exception as e:
        db = f"error: {str(e)[:100]}"
    return {"status": "healthy" if db == "connected" else "degraded", "database": db}


@app.get("/debug")
async def debug():
    routes = [
        {"path": r.path, "methods": sorted(r.methods or [])}
        for r in app.routes if hasattr(r, "methods")
    ]
    return {
        "router_errors": _errors,
        "routes_count":  len(routes),
        "routes":        routes,
        "env": {
            "GOOGLE_CLIENT_ID":     "SET" if os.environ.get("GOOGLE_CLIENT_ID") else "MISSING",
            "GOOGLE_CLIENT_SECRET": "SET" if os.environ.get("GOOGLE_CLIENT_SECRET") else "MISSING",
            "GOOGLE_REDIRECT_URI":  os.environ.get("GOOGLE_REDIRECT_URI", "MISSING"),
            "FRONTEND_URL":         os.environ.get("FRONTEND_URL", "MISSING"),
            "JWT_SECRET":           "SET" if os.environ.get("JWT_SECRET") else "MISSING",
            "GEMINI_API_KEY":       "SET" if os.environ.get("GEMINI_API_KEY") else "MISSING",
            "DATABASE_URL":         (os.environ.get("DATABASE_URL", "MISSING"))[:50],
        },
    }
