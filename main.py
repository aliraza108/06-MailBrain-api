"""
MailBrain API — main.py
Vercel Python serverless entry point.

Key design decisions for Vercel compatibility:
1. sys.path patched FIRST before any local imports
2. DB init is deferred — happens inside lifespan, NOT at module level
3. All routers imported inside try/except so one bad import never kills the app
4. Global exception handler returns JSON (not Vercel's HTML crash page)
5. /debug endpoint shows exactly what loaded and what env vars exist
"""
import os
import sys
import traceback
from contextlib import asynccontextmanager

# ── CRITICAL: patch path before ANY local imports ─────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations on cold start. Never let this crash the app."""
    try:
        from database import init_db
        await init_db()
        print("[MailBrain] ✅ DB tables ready")
    except Exception:
        print("[MailBrain] ❌ DB init error (app still starts):")
        traceback.print_exc()
    yield
    print("[MailBrain] shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "MailBrain API",
    description = "Autonomous AI Email Operations Manager",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Global error handler — always return JSON ─────────────────────────────────

@app.exception_handler(Exception)
async def _global_error(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[MailBrain] unhandled error @ {request.url}\n{tb}")
    return JSONResponse(
        status_code = 500,
        content     = {"error": str(exc), "path": str(request.url)},
    )


# ── Register routers (each in its own try/except) ─────────────────────────────

_router_errors: list[str] = []


def _load(module_path: str, attr: str, prefix: str, tag: str):
    try:
        import importlib
        mod    = importlib.import_module(module_path)
        router = getattr(mod, attr)
        app.include_router(router, prefix=prefix, tags=[tag])
        print(f"[MailBrain] ✅ {tag} router loaded")
    except Exception as e:
        msg = f"{tag}: {e}"
        _router_errors.append(msg)
        print(f"[MailBrain] ❌ {msg}")
        traceback.print_exc()


_load("routes.auth",      "router", "/auth",      "Auth")
_load("routes.emails",    "router", "/emails",    "Emails")
_load("routes.analytics", "router", "/analytics", "Analytics")
_load("routes.webhooks",  "router", "/webhooks",  "Webhooks")


# ── Base routes ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
async def root():
    return {
        "service": "MailBrain API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/health",
        "debug":   "/debug",
    }


@app.get("/health", tags=["Status"])
async def health():
    db_status = "unknown"
    try:
        from database import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"

    return {
        "status":         "healthy" if db_status == "connected" else "degraded",
        "database":       db_status,
        "router_errors":  _router_errors,
        "version":        "1.0.0",
    }


@app.get("/debug", tags=["Status"])
async def debug():
    """
    Shows loaded routes and env var status.
    Use this to diagnose deployment issues.
    """
    routes = [
        {"path": r.path, "methods": sorted(r.methods or [])}
        for r in app.routes
        if hasattr(r, "methods")
    ]
    env = {
        "GOOGLE_CLIENT_ID":     "✅ set" if os.environ.get("GOOGLE_CLIENT_ID") else "❌ MISSING",
        "GOOGLE_CLIENT_SECRET": "✅ set" if os.environ.get("GOOGLE_CLIENT_SECRET") else "❌ MISSING",
        "GOOGLE_REDIRECT_URI":  os.environ.get("GOOGLE_REDIRECT_URI", "❌ MISSING"),
        "FRONTEND_URL":         os.environ.get("FRONTEND_URL", "❌ MISSING"),
        "JWT_SECRET":           "✅ set" if os.environ.get("JWT_SECRET") else "❌ MISSING",
        "GEMINI_API_KEY":       "✅ set" if os.environ.get("GEMINI_API_KEY") else "❌ MISSING",
        "DATABASE_URL":         (os.environ.get("DATABASE_URL", "❌ MISSING") or "")[:50] + "…",
    }
    return {
        "router_errors": _router_errors,
        "routes_loaded": len(routes),
        "routes":        routes,
        "env":           env,
        "python":        sys.version,
        "sys_path_0":    sys.path[0],
    }