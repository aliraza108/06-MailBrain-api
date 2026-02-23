from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Google OAuth
    GOOGLE_CLIENT_ID: str = "221967337419-thv5q632u71op3sdn6lav51s3ncmq5r7.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str = "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"
    GOOGLE_REDIRECT_URI: str = "https://06-mailbrain-api.vercel.app/auth/callback"

    # Frontend
    FRONTEND_URL: str = "https://06-mailbrain.vercel.app"

    # JWT
    JWT_SECRET: str = "your-super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # AI (Gemini via OpenAI-compatible SDK)
    GEMINI_API_KEY: str = "AIzaSyAbIfJs--F8zeqtkIFHYmkxiEIBsDT0Dkk"
    AI_MODEL: str = "gemini-2.5-flash"
    AI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Automation
    AUTO_SEND_CONFIDENCE_THRESHOLD: float = 0.85  # Only auto-send if AI confidence >= this

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./mailbrain.db"

    # Gmail
    GMAIL_SCOPES: list = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()
