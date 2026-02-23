# MailBrain Backend API

Autonomous Email Operations Manager — FastAPI + Gemini AI + Gmail API

---

## 📁 Project Structure

```
mailbrain-backend/
├── main.py              # FastAPI app, CORS, route registration
├── database.py          # SQLAlchemy models (User, Email, ActivityLog)
├── config.py            # Settings via pydantic-settings + .env
├── ai_agent.py          # MailBrain AI Agent (openai-agents SDK → Gemini)
├── gmail_service.py     # Gmail API: fetch, parse, send emails
├── routes/
│   ├── auth.py          # Google OAuth2 flow + JWT
│   ├── emails.py        # Core email processing pipeline
│   ├── analytics.py     # Dashboard data endpoints
│   └── webhooks.py      # Gmail push notifications (Pub/Sub)
├── requirements.txt
├── vercel.json
└── .env.example
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your keys

# 3. Run locally
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## 🔑 Google OAuth — Authorized Redirect URIs

In **Google Cloud Console → APIs & Services → Credentials → OAuth 2.0 Client:**

Add these to **Authorized redirect URIs:**

```
https://06-mailbrain-api.vercel.app/auth/callback
http://localhost:8000/auth/callback
```

Add these to **Authorized JavaScript origins:**

```
https://06-mailbrain.vercel.app
https://06-mailbrain-api.vercel.app
http://localhost:3000
http://localhost:8000
```

---

## 🔌 API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/google` | Redirect to Google OAuth |
| GET | `/auth/callback` | OAuth callback → issues JWT |
| GET | `/auth/me` | Get current user |
| POST | `/auth/logout` | Logout |

### Emails
| Method | Path | Description |
|--------|------|-------------|
| GET | `/emails/` | List all processed emails (filterable) |
| POST | `/emails/sync` | Fetch & process Gmail unread emails |
| POST | `/emails/process` | Process a manually pasted email |
| POST | `/emails/batch` | Process multiple emails at once |
| GET | `/emails/{id}` | Get full email detail + AI analysis |
| POST | `/emails/{id}/approve` | Send AI-generated reply |
| POST | `/emails/{id}/reply` | Send custom reply |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics/overview` | Summary stats (totals, automation rate) |
| GET | `/analytics/intent` | Email distribution by intent |
| GET | `/analytics/priority` | Priority breakdown |
| GET | `/analytics/trends` | Daily volume trends |
| GET | `/analytics/automation` | Action and department routing stats |
| GET | `/analytics/escalations` | High-risk email list |

---

## 🧠 AI Pipeline

Every email goes through this pipeline:

```
Email In
   ↓
Intent Detection (10 categories)
   ↓
Priority Scoring (CRITICAL / HIGH / NORMAL / LOW + 0–1 score)
   ↓
Sentiment + Language Detection
   ↓
Action Decision (auto_reply / assign / ticket / meeting / flag)
   ↓
Reply Generation (tone: professional / empathetic / friendly / firm)
   ↓
Confidence Score
   ↓
Auto-send if confidence ≥ 0.85 AND action == auto_reply
   ↓
Logged to DB + Activity Log
```

---

## 🚢 Deploy to Vercel

```bash
npm i -g vercel
vercel login

# Set secrets
vercel env add GOOGLE_CLIENT_ID
vercel env add GOOGLE_CLIENT_SECRET
vercel env add JWT_SECRET
vercel env add GEMINI_API_KEY
vercel env add DATABASE_URL   # Use PostgreSQL for production!

vercel --prod
```

> **Production DB:** Replace SQLite with `postgresql+asyncpg://user:pass@host/mailbrain`  
> Recommended: Neon, Supabase, or Railway (all have free tiers)

---

## 🔒 Security Notes

- Never commit `.env` to git (it's in `.gitignore`)
- Change `JWT_SECRET` to a long random string in production
- Use PostgreSQL instead of SQLite in production
- Consider rotating `GEMINI_API_KEY` periodically
