import uuid
import httpx
import jwt
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, User
from config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI,
    FRONTEND_URL, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_HOURS, GMAIL_SCOPES
)

router = APIRouter()

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"


def create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(authorization: str = None, db: AsyncSession = Depends(get_db)):
    from fastapi import Header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/google")
async def google_login():
    scopes = "%20".join(GMAIL_SCOPES)
    url = (
        f"{GOOGLE_AUTH_URL}"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scopes}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url)


@router.get("/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    async with httpx.AsyncClient(timeout=30) as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })

        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp.text}")

        tokens = token_resp.json()

        user_resp = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_info = user_resp.json()

    # Upsert user
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
            token_expiry=expiry,
        )
        db.add(user)
    else:
        user.access_token  = tokens.get("access_token")
        user.refresh_token = tokens.get("refresh_token", user.refresh_token)
        user.token_expiry  = expiry
        user.name          = user_info.get("name")
        user.picture       = user_info.get("picture")

    await db.commit()
    await db.refresh(user)

    token = create_jwt(user.id, user.email)
    return RedirectResponse(f"{FRONTEND_URL}/dashboard?token={token}")


@router.get("/me")
async def get_me(authorization: str = None, db: AsyncSession = Depends(get_db)):
    from fastapi import Header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "email": user.email, "name": user.name, "picture": user.picture}


@router.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}