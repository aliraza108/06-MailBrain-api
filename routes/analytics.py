from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from database import get_db, User, Email
from routes.auth import verify_jwt

router = APIRouter()


async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    payload = verify_jwt(authorization.replace("Bearer ", ""))
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/overview")
async def overview(
    days: int = 7,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    base  = select(Email).where(Email.user_id == user.id, Email.processed_at >= since)

    total     = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    critical  = (await db.execute(select(func.count()).select_from(base.where(Email.priority == "CRITICAL").subquery()))).scalar() or 0
    auto_sent = (await db.execute(select(func.count()).select_from(base.where(Email.reply_sent == True).subquery()))).scalar() or 0
    escalated = (await db.execute(select(func.count()).select_from(base.where(Email.escalated == True).subquery()))).scalar() or 0
    avg_conf  = (await db.execute(
        select(func.avg(Email.confidence_score))
        .where(Email.user_id == user.id, Email.processed_at >= since)
    )).scalar() or 0

    return {
        "period_days": days,
        "total_emails": total,
        "critical_emails": critical,
        "auto_replied": auto_sent,
        "escalated": escalated,
        "automation_rate": round(auto_sent / total * 100 if total else 0, 1),
        "avg_confidence": round(float(avg_conf) * 100, 1),
    }


@router.get("/intent")
async def intent_distribution(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(Email.intent, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id, Email.processed_at >= since)
        .group_by(Email.intent).order_by(desc("count"))
    )).all()
    return {"distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]}


@router.get("/priority")
async def priority_breakdown(
    days: int = 7,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(Email.priority, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id, Email.processed_at >= since)
        .group_by(Email.priority)
    )).all()
    order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
    data  = sorted([{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows],
                   key=lambda x: order.get(x["priority"], 99))
    return {"breakdown": data}


@router.get("/trends")
async def trends(
    days: int = 14,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(func.date(Email.processed_at).label("date"), func.count(Email.id).label("total"))
        .where(Email.user_id == user.id, Email.processed_at >= since)
        .group_by(func.date(Email.processed_at))
        .order_by(func.date(Email.processed_at))
    )).all()
    return {"trends": [{"date": str(r.date), "total": r.total} for r in rows]}


@router.get("/automation")
async def automation(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Email.action_taken, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id)
        .group_by(Email.action_taken).order_by(desc("count"))
    )).all()
    total = sum(r.count for r in rows)
    return {
        "total_processed": total,
        "actions": [{"action": r.action_taken or "none", "count": r.count,
                     "percentage": round(r.count / total * 100, 1) if total else 0} for r in rows],
    }


@router.get("/escalations")
async def escalations(
    days: int = 7,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(Email)
        .where(Email.user_id == user.id, Email.processed_at >= since,
               (Email.escalated == True) | (Email.priority == "CRITICAL"))
        .order_by(desc(Email.priority_score))
    )).scalars().all()
    return {
        "count": len(rows),
        "emails": [{"id": e.id, "sender": e.sender, "subject": e.subject,
                    "priority": e.priority, "escalated": e.escalated,
                    "reply_sent": e.reply_sent, "processed_at": str(e.processed_at)} for e in rows],
    }