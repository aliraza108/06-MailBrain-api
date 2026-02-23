from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routes import auth, emails, analytics, webhooks
from database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="MailBrain API",
    description="Autonomous Email Operations Manager",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://06-mailbrain.vercel.app", "https://06-mailbrain.vercel.app", "http://localhost:3000"],
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(emails.router, prefix="/emails", tags=["Emails"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])

@app.get("/")
async def root():
    return {"message": "MailBrain API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
