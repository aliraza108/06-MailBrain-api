from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import traceback


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import init_db
        await init_db()
        print("✅ Startup complete")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
try:
    from routes import auth, emails, analytics, webhooks
    app.include_router(auth.router,      prefix="/auth",      tags=["Auth"])
    app.include_router(emails.router,    prefix="/emails",    tags=["Emails"])
    app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
    app.include_router(webhooks.router,  prefix="/webhooks",  tags=["Webhooks"])
    print("✅ Routes loaded")
except Exception as e:
    print(f"❌ Route loading failed: {e}")
    traceback.print_exc()


@app.get("/")
async def root():
    return {"message": "MailBrain API is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    try:
        from database import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": str(e)}