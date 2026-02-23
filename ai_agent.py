"""
MailBrain AI Agent
Uses openai-agents SDK pointed at Gemini's OpenAI-compatible endpoint.
"""

import json
import re
from typing import Optional
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled, AsyncOpenAI
from config import get_settings

settings = get_settings()

# ──────────────────────────────────────────────
# Configure Gemini as the AI backend
# ──────────────────────────────────────────────
client = AsyncOpenAI(
    base_url=settings.AI_BASE_URL,
    api_key=settings.GEMINI_API_KEY
)
set_default_openai_api("chat_completions")
set_default_openai_client(client=client)
set_tracing_disabled(True)


# ──────────────────────────────────────────────
# Tool Functions (called by the agent)
# ──────────────────────────────────────────────

def detect_intent(email_content: str) -> dict:
    """
    Analyze email and detect its primary intent.
    Returns intent category and confidence.
    """
    # This is called as a tool by the agent — the agent fills it via reasoning
    return {"status": "tool_called", "input": email_content}


def score_priority(intent: str, tone_keywords: str, subject: str) -> dict:
    """
    Score email urgency based on intent, tone keywords found, and subject.
    Returns priority level: CRITICAL | HIGH | NORMAL | LOW and numeric score 0-1.
    """
    return {"status": "tool_called", "intent": intent}


def decide_action(intent: str, priority: str, confidence: float) -> dict:
    """
    Decide the best action for this email.
    Actions: auto_reply | assign_department | create_ticket | schedule_meeting | flag_management | request_info
    Returns action type and relevant parameters.
    """
    return {"status": "tool_called"}


def generate_reply(intent: str, tone: str, original_email: str, sender_name: str) -> dict:
    """
    Generate a context-aware email reply.
    Tone options: professional | empathetic | friendly | firm
    Returns the complete reply text.
    """
    return {"status": "tool_called"}


# ──────────────────────────────────────────────
# The MailBrain Agent
# ──────────────────────────────────────────────

mailbrain_agent = Agent(
    name="MailBrain Email Operations Agent",
    instructions="""
You are MailBrain, an autonomous email operations AI. Your job is to fully analyze incoming emails
and produce a structured JSON analysis for each one.

For every email you receive, you MUST:

1. DETECT INTENT — Classify into exactly one of:
   support_request | refund_demand | sales_inquiry | meeting_request | complaint |
   spam | urgent_escalation | billing_question | partnership_offer | general_inquiry

2. SCORE PRIORITY — Assign exactly one of:
   CRITICAL (needs response <1h) | HIGH (needs response <4h) | NORMAL (<24h) | LOW (<72h)
   Also output a numeric score from 0.0 to 1.0 (1.0 = most urgent)

3. DETECT SENTIMENT — positive | neutral | negative

4. DETECT LANGUAGE — ISO 639-1 code (e.g., "en", "es", "fr", "ar")

5. DECIDE ACTION — Choose the best action:
   auto_reply | assign_department | create_ticket | schedule_meeting | flag_management | request_info
   Also specify department if assigning: support | billing | sales | management | technical

6. GENERATE REPLY — Write a complete, ready-to-send email reply.
   Choose tone based on context: professional | empathetic | friendly | firm
   Reply in the SAME LANGUAGE as the original email.

7. ASSESS CONFIDENCE — Your confidence in the analysis: 0.0 to 1.0

8. SUMMARIZE — One sentence summary of what the email is about.

9. ESCALATION RISK — Will this become a serious problem if not handled? true | false

10. FOLLOW-UP NEEDED — Should a follow-up reminder be set? true | false, and when (hours from now).

Output ONLY valid JSON in this exact format:
{
  "intent": "...",
  "priority": "...",
  "priority_score": 0.0,
  "sentiment": "...",
  "language": "...",
  "summary": "...",
  "action": "...",
  "assigned_department": "...",
  "confidence_score": 0.0,
  "escalation_risk": false,
  "follow_up_needed": false,
  "follow_up_hours": null,
  "reply_tone": "...",
  "generated_reply": "...",
  "keywords_detected": []
}
""",
    model=settings.AI_MODEL,
    tools=[detect_intent, score_priority, decide_action, generate_reply]
)


# ──────────────────────────────────────────────
# Main Analysis Function
# ──────────────────────────────────────────────

async def analyze_email(
    subject: str,
    body: str,
    sender: str,
    sender_name: str = "",
    thread_context: Optional[str] = None
) -> dict:
    """
    Run the MailBrain agent on a single email.
    Returns structured analysis dict.
    """
    prompt = f"""
Analyze this email completely:

FROM: {sender} ({sender_name or 'Unknown'})
SUBJECT: {subject}
BODY:
{body}
"""
    if thread_context:
        prompt += f"\n\nPREVIOUS THREAD CONTEXT:\n{thread_context}"

    try:
        result = await Runner.run(mailbrain_agent, input=prompt)
        output = result.final_output.strip()

        # Strip markdown code fences if present
        output = re.sub(r"^```json\s*", "", output)
        output = re.sub(r"\s*```$", "", output)

        analysis = json.loads(output)
        return {"success": True, "data": analysis}

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parse error: {str(e)}",
            "raw": result.final_output if 'result' in dir() else ""
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_batch(emails: list[dict]) -> list[dict]:
    """Analyze multiple emails concurrently."""
    import asyncio
    tasks = [
        analyze_email(
            subject=e.get("subject", ""),
            body=e.get("body", ""),
            sender=e.get("sender", ""),
            sender_name=e.get("sender_name", ""),
            thread_context=e.get("thread_context")
        )
        for e in emails
    ]
    return await asyncio.gather(*tasks)
