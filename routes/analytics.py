"""
Analytics Routes — Inbox Intelligence Dashboard
GET /analytics/overview       Summary stats
GET /analytics/intent         Email distribution by intent
GET /analytics/priority       Priority breakdown
GET /analytics/trends         Response time and volume trends
GET /analytics/automation     Automation success rate
GET /analytics/escalations    Escalation risk tracking
"""
from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta

from database import get_db, Email, ActivityLog, User
from routes.auth import verify_jwt

router = APIRouter()


async def get_user_from_token(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException
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
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Summary stats for the dashboard."""
    since = datetime.utcnow() - timedelta(days=days)
    base = select(Email).where(Email.user_id == user.id, Email.processed_at >= since)

    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar()

    # Critical emails
    critical_q = base.where(Email.priority == "CRITICAL")
    critical = (await db.execute(select(func.count()).select_from(critical_q.subquery()))).scalar()

    # Auto-replied
    auto_q = base.where(Email.reply_sent == True)
    auto_replied = (await db.execute(select(func.count()).select_from(auto_q.subquery()))).scalar()

    # Escalated
    esc_q = base.where(Email.escalated == True)
    escalated = (await db.execute(select(func.count()).select_from(esc_q.subquery()))).scalar()

    # Avg confidence
    avg_conf_q = select(func.avg(Email.confidence_score)).where(
        Email.user_id == user.id, Email.processed_at >= since
    )
    avg_confidence = (await db.execute(avg_conf_q)).scalar() or 0

    return {
        "period_days": days,
        "total_emails": total,
        "critical_emails": critical,
        "auto_replied": auto_replied,
        "escalated": escalated,
        "automation_rate": round((auto_replied / total * 100) if total > 0 else 0, 1),
        "avg_confidence": round(avg_confidence * 100, 1)
    }


@router.get("/intent")
async def intent_distribution(
    days: int = 30,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Email distribution by detected intent."""
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(Email.intent, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id, Email.processed_at >= since)
        .group_by(Email.intent)
        .order_by(desc("count"))
    )
    rows = result.all()
    return {
        "distribution": [{"intent": r.intent or "unknown", "count": r.count} for r in rows]
    }


@router.get("/priority")
async def priority_breakdown(
    days: int = 7,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Priority level breakdown."""
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(Email.priority, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id, Email.processed_at >= since)
        .group_by(Email.priority)
    )
    rows = result.all()
    order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
    data = sorted(
        [{"priority": r.priority or "UNKNOWN", "count": r.count} for r in rows],
        key=lambda x: order.get(x["priority"], 99)
    )
    return {"breakdown": data}


@router.get("/trends")
async def volume_trends(
    days: int = 14,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Daily email volume for the past N days."""
    result = await db.execute(
        select(
            func.date(Email.processed_at).label("date"),
            func.count(Email.id).label("total"),
            func.sum(func.cast(Email.reply_sent, int)).label("auto_replied"),
            func.sum(func.cast(Email.escalated, int)).label("escalated")
        )
        .where(
            Email.user_id == user.id,
            Email.processed_at >= datetime.utcnow() - timedelta(days=days)
        )
        .group_by(func.date(Email.processed_at))
        .order_by(func.date(Email.processed_at))
    )
    rows = result.all()
    return {
        "trends": [
            {
                "date": str(r.date),
                "total": r.total,
                "auto_replied": r.auto_replied or 0,
                "escalated": r.escalated or 0
            }
            for r in rows
        ]
    }


@router.get("/automation")
async def automation_stats(
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """Automation action breakdown and success rate."""
    result = await db.execute(
        select(Email.action_taken, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id)
        .group_by(Email.action_taken)
        .order_by(desc("count"))
    )
    rows = result.all()

    total = sum(r.count for r in rows)
    actions = [{"action": r.action_taken or "none", "count": r.count,
                "percentage": round(r.count / total * 100, 1) if total else 0} for r in rows]

    # Department distribution
    dept_result = await db.execute(
        select(Email.assigned_department, func.count(Email.id).label("count"))
        .where(Email.user_id == user.id, Email.assigned_department != None)
        .group_by(Email.assigned_department)
    )
    dept_rows = dept_result.all()

    return {
        "total_processed": total,
        "actions": actions,
        "department_routing": [{"department": r.assigned_department, "count": r.count} for r in dept_rows]
    }


@router.get("/escalations")
async def escalation_report(
    days: int = 7,
    user: User = Depends(get_user_from_token),
    db: AsyncSession = Depends(get_db)
):
    """List high-risk and escalated emails."""
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(Email)
        .where(
            Email.user_id == user.id,
            Email.processed_at >= since,
            (Email.escalated == True) | (Email.priority == "CRITICAL")
        )
        .order_by(desc(Email.priority_score))
    )
    emails = result.scalars().all()
    return {
        "count": len(emails),
        "emails": [
            {
                "id": e.id,
                "sender": e.sender,
                "subject": e.subject,
                "summary": e.summary,
                "priority": e.priority,
                "intent": e.intent,
                "escalated": e.escalated,
                "reply_sent": e.reply_sent,
                "processed_at": e.processed_at
            }
            for e in emails
        ]
    }
