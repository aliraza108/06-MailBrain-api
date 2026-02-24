import os

# Google OAuth
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "https://06-mailbrain-api.vercel.app/auth/callback")

# Frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://06-mailbrain.vercel.app")

# JWT
JWT_SECRET     = os.getenv("JWT_SECRET", "changeme")
JWT_ALGORITHM  = "HS256"
JWT_EXPIRE_HOURS = 24

# AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AI_MODEL       = os.getenv("AI_MODEL", "gemini-2.5-flash")
AI_BASE_URL    = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Automation
AUTO_SEND_THRESHOLD = float(os.getenv("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))

# Gmail
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]