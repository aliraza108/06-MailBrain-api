from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import EmailRecord, User, get_db
from routes.auth import decode_token

router = APIRouter()


async def require_user(
    authorization: str = Header(None),
    db: AsyncSession  = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <token>")
    payload = decode_token(authorization[7:].strip())
    result  = await db.execute(select(User).where(User.id == payload["sub"]))
    user    = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return user


@router.get("/overview")
async def overview(
    days: int          = 7,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    w = [EmailRecord.user_id == user.id, EmailRecord.processed_at >= since]

    total    = (await db.execute(select(func.count(EmailRecord.id)).where(*w))).scalar() or 0
    critical = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.priority == "CRITICAL"))).scalar() or 0
    auto_r   = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.reply_sent == True))).scalar() or 0
    esc      = (await db.execute(select(func.count(EmailRecord.id)).where(*w, EmailRecord.escalated == True))).scalar() or 0
    avg_c    = (await db.execute(select(func.avg(EmailRecord.confidence_score)).where(*w))).scalar() or 0

    return {
        "period_days":     days,
        "total_emails":    total,
        "critical_emails": critical,
        "auto_replied":    auto_r,
        "escalated":       esc,
        "automation_rate": round(auto_r / total * 100 if total else 0, 1),
        "avg_confidence":  round(float(avg_c) * 100, 1),
    }


@router.get("/intent")
async def intent_dist(
    days: int          = 30,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord.intent, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(EmailRecord.intent)
        .order_by(desc("count"))
    )).all()
    return {"distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]}


@router.get("/priority")
async def priority_breakdown(
    days: int          = 7,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord.priority, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(EmailRecord.priority)
    )).all()
    order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
    return {"breakdown": sorted(
        [{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows],
        key=lambda x: order.get(x["priority"], 9),
    )}


@router.get("/trends")
async def trends(
    days: int          = 14,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(
            func.date(EmailRecord.processed_at).label("date"),
            func.count(EmailRecord.id).label("total"),
        )
        .where(EmailRecord.user_id == user.id, EmailRecord.processed_at >= since)
        .group_by(func.date(EmailRecord.processed_at))
        .order_by(func.date(EmailRecord.processed_at))
    )).all()
    return {"trends": [{"date": str(r.date), "total": r.total} for r in rows]}


@router.get("/automation")
async def automation(
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(EmailRecord.action_taken, func.count(EmailRecord.id).label("count"))
        .where(EmailRecord.user_id == user.id)
        .group_by(EmailRecord.action_taken)
        .order_by(desc("count"))
    )).all()
    total = sum(r.count for r in rows)
    return {
        "total_processed": total,
        "actions": [{
            "action":     r.action_taken or "none",
            "count":      r.count,
            "percentage": round(r.count / total * 100, 1) if total else 0,
        } for r in rows],
    }


@router.get("/escalations")
async def escalations(
    days: int          = 7,
    user: User         = Depends(require_user),
    db:   AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (await db.execute(
        select(EmailRecord)
        .where(
            EmailRecord.user_id == user.id,
            EmailRecord.processed_at >= since,
            (EmailRecord.escalated == True) | (EmailRecord.priority == "CRITICAL"),
        )
        .order_by(desc(EmailRecord.priority_score))
    )).scalars().all()
    return {
        "count":  len(rows),
        "emails": [{
            "id":           e.id,
            "sender":       e.sender,
            "subject":      e.subject,
            "summary":      e.summary,
            "priority":     e.priority,
            "intent":       e.intent,
            "escalated":    e.escalated,
            "reply_sent":   e.reply_sent,
            "processed_at": str(e.processed_at),
        } for e in rows],
    }
