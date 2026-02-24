import os

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "https://06-mailbrain-api.vercel.app/auth/callback")
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "https://06-mailbrain.vercel.app")
JWT_SECRET           = os.environ.get("JWT_SECRET", "dev-secret-change-in-prod")
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
GMAIL_API_BASE   = "https://gmail.googleapis.com/gmail/v1"

GMAIL_SCOPES = (
    "openid "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile "
    "https://www.googleapis.com/auth/gmail.readonly "
    "https://www.googleapis.com/auth/gmail.send "
    "https://www.googleapis.com/auth/gmail.modify"
)
