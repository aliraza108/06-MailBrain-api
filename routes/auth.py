"""
Google OAuth2 Authentication Routes
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import httpx
import jwt
import uuid
from datetime import datetime, timedelta

from database import get_db, User
from config import get_settings

settings = get_settings()
router = APIRouter()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(token: str = None, db: AsyncSession = Depends(get_db)):
    from fastapi import Header
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    payload = verify_jwt(token)
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get("/google")
async def google_login():
    """Redirect user to Google OAuth consent screen."""
    scopes = " ".join(settings.GMAIL_SCOPES)
    params = (
        f"client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scopes}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")


@router.get("/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Handle Google OAuth callback, exchange code for tokens."""
    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        })

        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code for tokens")

        tokens = token_resp.json()

        # Get user info
        user_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        user_info = user_resp.json()

    # Save/update user in DB
    result = await db.execute(select(User).where(User.email == user_info["email"]))
    user = result.scalar_one_or_none()

    expiry = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))

    if not user:
        user = User(
            id=str(uuid.uuid4()),
            email=user_info["email"],
            name=user_info.get("name"),
            picture=user_info.get("picture"),
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_expiry=expiry
        )
        db.add(user)
    else:
        user.access_token = tokens.get("access_token")
        user.refresh_token = tokens.get("refresh_token", user.refresh_token)
        user.token_expiry = expiry
        user.name = user_info.get("name")
        user.picture = user_info.get("picture")

    await db.commit()
    await db.refresh(user)

    # Create JWT and redirect to frontend
    jwt_token = create_jwt(user.id, user.email)
    return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?token={jwt_token}")


@router.get("/me")
async def get_me(authorization: str = None, db: AsyncSession = Depends(get_db)):
    """Get current authenticated user info."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = authorization.replace("Bearer ", "")
    payload = verify_jwt(token)

    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture
    }


@router.post("/refresh")
async def refresh_token(user_id: str, db: AsyncSession = Depends(get_db)):
    """Refresh Google access token using stored refresh token."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.refresh_token:
        raise HTTPException(status_code=400, detail="No refresh token available")

    async with httpx.AsyncClient() as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data={
            "refresh_token": user.refresh_token,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "grant_type": "refresh_token"
        })
        tokens = resp.json()

    user.access_token = tokens.get("access_token")
    user.token_expiry = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))
    await db.commit()

    return {"access_token": user.access_token}


@router.post("/logout")
async def logout():
    """Client should discard JWT token."""
    return {"message": "Logged out successfully"}
