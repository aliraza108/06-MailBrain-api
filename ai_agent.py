"""
MailBrain AI — Gemini analysis via direct REST calls.
No openai/openai-agents SDK dependency.
Always returns a result — falls back to rule-based if AI fails.
"""
import json
import re
import httpx
from config import GEMINI_API_KEY, AI_MODEL, AI_BASE_URL


_PROMPT = """\
Analyze this email and return ONLY a JSON object. No explanation, no markdown, no backticks.

FROM: {sender}
SUBJECT: {subject}
BODY:
{body}
{thread_part}

Required JSON (every field is mandatory):
{{
  "intent": "support_request|refund_demand|sales_inquiry|meeting_request|complaint|spam|urgent_escalation|billing_question|partnership_offer|general_inquiry",
  "priority": "CRITICAL|HIGH|NORMAL|LOW",
  "priority_score": 0.85,
  "sentiment": "positive|neutral|negative",
  "language": "en",
  "summary": "One sentence summary.",
  "action": "auto_reply|assign_department|create_ticket|schedule_meeting|flag_management|request_info",
  "assigned_department": "support|billing|sales|management|technical|none",
  "confidence_score": 0.90,
  "escalation_risk": false,
  "follow_up_needed": false,
  "follow_up_hours": null,
  "reply_tone": "professional|empathetic|friendly|firm",
  "generated_reply": "Dear [Name],\\n\\nFull reply text here...\\n\\nBest regards,\\nSupport Team",
  "keywords_detected": ["word1", "word2"]
}}"""


def _fallback(subject: str, body: str) -> dict:
    """Rule-based analysis used when Gemini is unavailable."""
    text = (subject + " " + body).lower()

    if any(w in text for w in ["urgent", "asap", "emergency", "immediately", "critical"]):
        intent, priority, dept, score = "urgent_escalation", "CRITICAL", "management", 0.95
    elif any(w in text for w in ["refund", "money back", "reimburse", "chargeback"]):
        intent, priority, dept, score = "refund_demand", "HIGH", "billing", 0.80
    elif any(w in text for w in ["complaint", "unacceptable", "terrible", "disgusting", "awful"]):
        intent, priority, dept, score = "complaint", "HIGH", "support", 0.75
    elif any(w in text for w in ["invoice", "billing", "payment", "subscription", "charge"]):
        intent, priority, dept, score = "billing_question", "HIGH", "billing", 0.75
    elif any(w in text for w in ["meeting", "schedule", "call", "demo", "appointment"]):
        intent, priority, dept, score = "meeting_request", "NORMAL", "sales", 0.70
    elif any(w in text for w in ["partner", "collaboration", "integrate", "api", "business"]):
        intent, priority, dept, score = "partnership_offer", "LOW", "sales", 0.60
    elif any(w in text for w in ["unsubscribe", "spam", "promotional", "offer"]):
        intent, priority, dept, score = "spam", "LOW", "none", 0.50
    elif any(w in text for w in ["bug", "error", "broken", "not working", "issue"]):
        intent, priority, dept, score = "support_request", "HIGH", "technical", 0.75
    else:
        intent, priority, dept, score = "general_inquiry", "NORMAL", "support", 0.60

    negative_words = ["angry", "furious", "terrible", "awful", "hate", "worst", "disgusting"]
    sentiment = "negative" if any(w in text for w in negative_words) else "neutral"

    return {
        "intent":              intent,
        "priority":            priority,
        "priority_score":      score,
        "sentiment":           sentiment,
        "language":            "en",
        "summary":             f"Email about: {subject[:100]}",
        "action":              "assign_department",
        "assigned_department": dept,
        "confidence_score":    0.60,
        "escalation_risk":     priority in ("CRITICAL", "HIGH"),
        "follow_up_needed":    priority == "CRITICAL",
        "follow_up_hours":     2 if priority == "CRITICAL" else None,
        "reply_tone":          "empathetic" if sentiment == "negative" else "professional",
        "generated_reply": (
            f"Dear Customer,\n\nThank you for contacting us regarding \"{subject}\".\n"
            f"We have received your message and our {dept} team will respond within 24 hours.\n\n"
            f"Best regards,\nMailBrain Support Team"
        ),
        "keywords_detected": [],
    }


def _clean_json(raw: str) -> str:
    """Strip markdown fences and whitespace from AI output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _fill_defaults(data: dict) -> dict:
    """Ensure every required key exists."""
    defaults = {
        "intent": "general_inquiry",
        "priority": "NORMAL",
        "priority_score": 0.5,
        "sentiment": "neutral",
        "language": "en",
        "summary": "Email received.",
        "action": "auto_reply",
        "assigned_department": "support",
        "confidence_score": 0.7,
        "escalation_risk": False,
        "follow_up_needed": False,
        "follow_up_hours": None,
        "reply_tone": "professional",
        "generated_reply": "Thank you for your email. We will respond shortly.",
        "keywords_detected": [],
    }
    for k, v in defaults.items():
        if k not in data or data[k] is None:
            data[k] = v
    return data


async def analyze_email(
    subject: str,
    body: str,
    sender: str,
    thread_context: str = None,
) -> dict:
    """
    Analyze an email with Gemini AI.
    Always returns {"success": bool, "data": {...}}.
    Never raises — falls back gracefully.
    """
    if not GEMINI_API_KEY:
        return {"success": False, "error": "No GEMINI_API_KEY", "data": _fallback(subject, body)}

    thread_part = f"\nPREVIOUS THREAD:\n{thread_context[:600]}" if thread_context else ""
    prompt = _PROMPT.format(
        sender=sender or "unknown@email.com",
        subject=subject or "(no subject)",
        body=(body or "")[:3000],
        thread_part=thread_part,
    )

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{AI_BASE_URL}chat/completions",
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model":       AI_MODEL,
                    "max_tokens":  2048,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are MailBrain. Return ONLY valid JSON. No markdown. No explanations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        cleaned = _clean_json(content)
        parsed  = json.loads(cleaned)
        data    = _fill_defaults(parsed)
        return {"success": True, "data": data}

    except json.JSONDecodeError:
        # AI returned non-JSON — use fallback but mark partial success
        return {"success": False, "error": "invalid_json", "data": _fallback(subject, body)}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"gemini_http_{e.response.status_code}", "data": _fallback(subject, body)}
    except httpx.TimeoutException:
        return {"success": False, "error": "gemini_timeout", "data": _fallback(subject, body)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": _fallback(subject, body)}
