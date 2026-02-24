import uuid
from datetime import datetime, timedelta

import httpx
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    GOOGLE_AUTH_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI, GOOGLE_TOKEN_URL, GOOGLE_USER_URL,
    FRONTEND_URL, GMAIL_SCOPES,
    JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET,
)
from database import User, get_db

router = APIRouter()


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(user_id: str, email: str) -> str:
    return jwt.encode(
        {
            "sub":   user_id,
            "email": email,
            "exp":   datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
            "iat":   datetime.utcnow(),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired — please login again")
    except Exception:
        raise HTTPException(401, "Invalid token")


# ── Reusable dependency ───────────────────────────────────────────────────────

async def require_user(
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <token>")
    payload = decode_token(authorization[7:].strip())
    result  = await db.execute(select(User).where(User.id == payload["sub"]))
    user    = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found — please login again")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/google", summary="Redirect to Google OAuth")
async def login():
    """Start the Google OAuth2 flow."""
    url = (
        f"{GOOGLE_AUTH_URL}"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={GMAIL_SCOPES.replace(' ', '%20')}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url=url, status_code=302)


@router.get("/callback", summary="Google OAuth callback")
async def callback(code: str, db: AsyncSession = Depends(get_db)):
    """
    Google redirects here with ?code=...
    Exchange for tokens → upsert user → issue JWT → redirect to frontend.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Exchange auth code for access/refresh tokens
        tok = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
        )
        if tok.status_code != 200:
            raise HTTPException(400, f"Token exchange failed: {tok.text[:300]}")
        tokens = tok.json()

        # 2. Fetch Google profile
        prof = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        profile = prof.json()

    # 3. Upsert user
    result = await db.execute(select(User).where(User.email == profile["email"]))
    user   = result.scalar_one_or_none()
    expiry = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))

    if user:
        user.access_token  = tokens["access_token"]
        user.refresh_token = tokens.get("refresh_token", user.refresh_token)
        user.token_expiry  = expiry
        user.name          = profile.get("name")
        user.picture       = profile.get("picture")
    else:
        user = User(
            id=str(uuid.uuid4()),
            email=profile["email"],
            name=profile.get("name"),
            picture=profile.get("picture"),
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            token_expiry=expiry,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)

    # 4. Issue JWT and send user to frontend dashboard
    token = create_token(user.id, user.email)
    return RedirectResponse(
        url=f"{FRONTEND_URL}/dashboard?token={token}",
        status_code=302,
    )


@router.get("/me", summary="Get current user")
async def me(user: User = Depends(require_user)):
    return {
        "id":      user.id,
        "email":   user.email,
        "name":    user.name,
        "picture": user.picture,
    }


@router.post("/logout", summary="Logout")
async def logout():
    return {"message": "Delete your JWT client-side to logout"}