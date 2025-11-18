import os
import re
import json
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

import numpy as np
import requests
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# ------------ ENV / CONFIG ------------

load_dotenv()

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

app = Flask(__name__)

print(">>> chat_app.py loaded with platform cleaner + custom UI <<<")

# ------------ HTML TEMPLATE (CUSTOM UI) ------------

FORM_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Live Broadcasting Chatbot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
      :root {
        --bg-dark: #020617;
        --bg-card: rgba(15, 23, 42, 0.92);
        --border-subtle: rgba(148, 163, 184, 0.35);
        --accent: #22c55e;
        --accent-soft: rgba(34, 197, 94, 0.18);
        --accent-strong: #22c55e;
        --text-main: #e5e7eb;
        --text-soft: #9ca3af;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text-main);
        background:
          radial-gradient(circle at top left, #1d293b 0, transparent 55%),
          radial-gradient(circle at bottom right, #0f172a 0, transparent 55%),
          linear-gradient(135deg, #020617, #020617);
      }

      .page-shell {
        width: 100%;
        max-width: 1024px;
        display: grid;
        gap: 16px;
        grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
      }

      @media (max-width: 768px) {
        .page-shell {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      .card {
        background: var(--bg-card);
        border-radius: 24px;
        padding: 20px 22px;
        border: 1px solid var(--border-subtle);
        box-shadow:
          0 24px 60px rgba(0, 0, 0, 0.55),
          0 0 0 1px rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(18px);
      }

      .header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
      }

      .logo-badge {
        width: 40px;
        height: 40px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 30% 20%, #4ade80, #16a34a);
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.7);
        font-weight: 700;
        color: #022c22;
        font-size: 18px;
      }

      .header-text h1 {
        font-size: 18px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .live-pill {
        padding: 2px 10px;
        font-size: 11px;
        border-radius: 999px;
        border: 1px solid rgba(248, 113, 113, 0.6);
        color: #fecaca;
        background: rgba(127, 29, 29, 0.45);
      }

      .header-text p {
        margin: 2px 0 0;
        font-size: 13px;
        color: var(--text-soft);
      }

      .query-form {
        margin-top: 12px;
        display: flex;
        flex-direction: row;
        gap: 10px;
      }

      @media (max-width: 640px) {
        .query-form {
          flex-direction: column;
        }
      }

      .query-input-wrap {
        position: relative;
        flex: 1;
      }

      .query-input-wrap input {
        width: 100%;
        padding: 11px 38px 11px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background: rgba(15, 23, 42, 0.85);
        color: var(--text-main);
        font-size: 14px;
        outline: none;
      }

      .query-input-wrap input::placeholder {
        color: #6b7280;
      }

      .query-input-wrap span {
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        color: #9ca3af;
      }

      .submit-btn {
        border: none;
        border-radius: 999px;
        padding: 0 20px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #022c22;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        white-space: nowrap;
      }

      .submit-btn:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
      }

      .submit-btn span.icon {
        font-size: 15px;
      }

      .hint-strip {
        margin-top: 10px;
        font-size: 11px;
        color: var(--text-soft);
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .hint-chip {
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.85);
      }

      .chat-block {
        margin-top: 18px;
        border-radius: 18px;
        padding: 14px 16px;
        background: radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 60%),
                    rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.5);
      }

      .chat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9ca3af;
        margin-bottom: 6px;
      }

      .user-question {
        font-size: 13px;
        margin-bottom: 10px;
        color: var(--text-soft);
      }

      .user-question strong {
        color: var(--text-main);
      }

      .bot-bubble {
        margin: 0;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-size: 14px;
        line-height: 1.5;
      }

      .tagline {
        margin-top: 10px;
        font-size: 11px;
        color: var(--text-soft);
      }

      .tagline strong {
        color: var(--accent-strong);
      }

      /* Right column: sources + status */
      .side-card-title {
        font-size: 13px;
        font-weight: 600;
        margin: 0 0 8px;
      }

      .mode-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 11px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background: rgba(15, 23, 42, 0.9);
        margin-bottom: 10px;
      }

      .mode-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.9);
      }

      .sources-list {
        list-style: none;
        margin: 8px 0 0;
        padding: 0;
        max-height: 210px;
        overflow: auto;
        font-size: 12px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(55, 65, 81, 0.9);
      }

      .sources-list li {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(31, 41, 55, 0.95);
      }

      .sources-list li:last-child {
        border-bottom: none;
      }

      .sources-list small {
        color: var(--text-soft);
      }

      .help-text {
        margin-top: 10px;
        font-size: 11px;
        color: var(--text-soft);
      }

      .contact-box {
        margin-top: 12px;
        padding: 10px 11px;
        border-radius: 14px;
        background: var(--accent-soft);
        color: #bbf7d0;
        font-size: 12px;
        border: 1px solid rgba(34, 197, 94, 0.6);
      }

      .contact-box strong {
        color: #ecfdf5;
      }

      .contact-box .phone {
        display: block;
        margin-top: 4px;
      }
    </style>
  </head>

  <body>
    <div class="page-shell">

      <!-- LEFT: main chat card -->
      <section class="card">
        <header class="header">
          <div class="logo-badge">LB</div>
          <div class="header-text">
            <h1>
              Live Broadcasting Assistant
              <span class="live-pill">LIVE READY</span>
            </h1>
            <p>Ask about services, events, live streaming, packages and more.</p>
          </div>
        </header>

        <form method="post" class="query-form">
          <div class="query-input-wrap">
            <input
              name="query"
              placeholder="Example: I need a 5-camera setup for a cricket match"
              value="{{ request.form.get('query','') }}"
              autocomplete="off"
            >
            <span>Ask</span>
          </div>
          <button type="submit" class="submit-btn">
            <span class="icon">➤</span>
            Ask now
          </button>
        </form>

        <div class="hint-strip">
          <span class="hint-chip">“What services do you offer?”</span>
          <span class="hint-chip">“Can you stream our wedding live?”</span>
          <span class="hint-chip">“Do you cover football matches?”</span>
          <span class="hint-chip">“I want a quote for a live event”</span>
        </div>

        {% if answer %}
        <div class="chat-block">
          <div class="chat-label">Latest reply</div>

          {% if request.form.get('query') %}
          <div class="user-question">
            <strong>You:</strong> {{ request.form.get('query') }}
          </div>
          {% endif %}

          <p class="bot-bubble">
            {{ answer|replace('\\n', '<br>')|safe }}
          </p>

          <p class="tagline">
            This assistant answers using your website content, FAQs and smart matching.
            <strong>It does not invent random services or prices.</strong>
          </p>
        </div>
        {% endif %}
      </section>

      <!-- RIGHT: status + sources + contact -->
      <aside class="card">
        <p class="side-card-title">How this assistant is answering</p>

        {% if mode_note %}
        <div class="mode-pill">
          <span class="mode-dot"></span>
          <span>{{ mode_note }}</span>
        </div>
        {% else %}
        <div class="mode-pill">
          <span class="mode-dot"></span>
          <span>Waiting for your first question…</span>
        </div>
        {% endif %}

        {% if sources and sources|length > 0 %}
        <ul class="sources-list">
          {% for s in sources %}
          <li>
            <small>{{ s }}</small>
          </li>
          {% endfor %}
        </ul>
        {% else %}
        <div class="help-text">
          After you ask something, this panel will show which part of your website/FAQ
          the answer was based on.
        </div>
        {% endif %}

        <div class="contact-box">
          <strong>Need an exact quote?</strong>
          <span class="phone">📞 +91-11-42908809 / +91-9911013303</span>
          <span class="phone">📝 Or fill the enquiry form on the website.</span>
        </div>
      </aside>

    </div>
  </body>
</html>
"""

# ------------ LOAD WEBSITE TEXT ------------

RAW_TEXT_PATH = "page_text.txt"

if os.path.exists(RAW_TEXT_PATH):
    with open(RAW_TEXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
else:
    raw_text = ""


def split_into_snippets(text: str) -> List[str]:
    parts = re.split(r"\\n\\s*\\n", text)
    return [p.strip() for p in parts if p.strip()]


# ------------ LOAD FAQ JSON ------------

def load_faqs(path: str = "faq.json") -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    faqs: List[Tuple[str, str]] = []
    for item in data:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue
        full = f"Q: {q}\\nA: {a}"
        faqs.append((q.lower(), full))
    return faqs


FAQ_PAIRS: List[Tuple[str, str]] = load_faqs()

# Manual FAQ for services
SERVICES_FAQ_FULL = (
    "Q: What services do you offer?\\n"
    "A: We provide complete live broadcasting solutions including multi-cam production, "
    "simulcast streaming, adaptive bitrate streaming, Instagram/Facebook/YouTube Live, "
    "video editing, VOD, 360° live, wedding streaming, sports broadcasting, "
    "corporate events, government events, religious streaming and more."
)
FAQ_PAIRS.append(("what services do you offer?", SERVICES_FAQ_FULL))

# ------------ WEBSITE SNIPPETS ------------

snippets: List[str] = split_into_snippets(raw_text)

EXTRA_INFO = (
    "Head Office: A-92 C/2, 4th Floor, Nambardar Estate, New Friends Colony, New Delhi, India. "
    "Phone: +91-11-42908809, +91-9911013303. Email: info@netnovaz.com."
)
snippets.append(EXTRA_INFO)

# ------------ TF-IDF ------------

if snippets:
    tfidf_vectorizer = TfidfVectorizer().fit(snippets)
    tfidf_matrix = tfidf_vectorizer.transform(snippets)
else:
    tfidf_vectorizer = None
    tfidf_matrix = None


def simple_retrieval(q: str, k: int = 3):
    if not tfidf_vectorizer or tfidf_matrix is None:
        return []
    q_vec = tfidf_vectorizer.transform([q])
    sims = (tfidf_matrix @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(-sims)[:k]

    out = []
    for idx in top_idx:
        if sims[idx] > 0:
            out.append(snippets[int(idx)])
    return out


# ------------ COMMON UTILS ------------

STOPWORDS = {
    "what", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did",
    "you", "your", "yours", "we", "our", "ours",
    "can", "could", "will", "would", "shall", "should",
    "a", "an", "the", "to", "for", "of", "in", "on", "and",
    "or", "with", "from", "about", "it", "this", "that"
}

SYNONYMS = {
    "stream": ["streaming", "live stream", "broadcast", "broadcasting", "telecast"],
    "camera": ["cam", "cams", "setup"],
    "football": ["soccer", "match", "sports"],
}


def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def expand_synonyms(q: str) -> str:
    words = q.split()
    expanded = words[:]
    for w in words:
        if w in SYNONYMS:
            expanded.extend(SYNONYMS[w])
    return " ".join(expanded)


# ------------ PLATFORM WORD CLEANER ------------

def clean_answer(text: str) -> str:
    """
    Replace 'platform' wording with 'broadcasting' so it sounds like a service,
    not a software platform.
    """
    if not text:
        return text

    replacements = {
        "platforms": "broadcasting services",
        "platform": "broadcasting",
        "Platform": "Broadcasting",
        "PLATFORM": "BROADCASTING",
    }

    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


# ------------ PRICING RULE ------------

def pricing_answer(question: str) -> Optional[str]:
    q = question.lower()

    pricing_keywords = [
        "price", "pricing", "cost", "charges", "charge", "rate", "rates",
        "quotation", "quote", "budget", "fees", "fee", "package",
        "how much", "per day", "per match", "per hour", "per event",
        "estimate", "approx price", "rough idea"
    ]

    sloppy_keywords = [
        "quat", "quatation", "qout", "qotation", "qoute", "quation"
    ]

    if any(w in q for w in pricing_keywords) or any(w in q for w in sloppy_keywords):
        msg = (
            "For pricing and custom quotations (e.g. a 5-camera cricket setup), "
            "please contact our team:\\n\\n"
            "📞 Call: +91-11-42908809 / +91-9911013303\\n"
            "📝 Or fill the enquiry form on our website.\\n\\n"
            "Once we know your exact requirements (camera count, duration, city, platforms, "
            "graphics, replays, etc.) we’ll send a tailored quote."
        )
        return clean_answer(msg)

    return None


# ------------ FAQ MATCHING WITH SYNONYMS ------------

def faq_match(question: str) -> Optional[str]:
    if not FAQ_PAIRS:
        return None

    q_norm = normalize(question)
    q_expanded = expand_synonyms(q_norm)

    best_score = 0.0
    best_ans = None

    for fq, full in FAQ_PAIRS:
        fq_norm = normalize(fq)

        # direct contains
        if fq_norm in q_expanded or q_expanded in fq_norm:
            return clean_answer(full)

        r = SequenceMatcher(None, q_expanded, fq_norm).ratio()
        if r > best_score:
            best_score = r
            best_ans = full

    if best_score >= 0.56 and best_ans:
        return clean_answer(best_ans)
    return None


# ------------ FALLBACK WEBSITE ANSWER ------------

def simple_answer(question: str, retrieved: List[str]):
    if retrieved:
        body = (
            "Here’s what we found related to your question:\\n\\n"
            + "\\n\\n".join(retrieved)
            + "\\n\\nFor a quick quote — call +91-11-42908809 / +91-9911013303 "
              "or fill the enquiry form on our website."
        )
        return clean_answer(body), retrieved, "Website text match"
    return (
        clean_answer("Sorry, nothing found on the site."),
        [],
        "No match",
    )


# ------------ GROQ AI CALL ------------

def call_groq_llm(q: str, retrieved: List[str]) -> Optional[str]:
    if not GROQ_API_KEY or not retrieved:
        return None

    context = ""
    for i, t in enumerate(retrieved, 1):
        context += f"SOURCE {i}:\\n{t}\\n\\n"

    prompt = (
        "Answer ONLY using this website content.\\n"
        "At the end, say exactly:\\n"
        "'For a quick quote — call +91-11-42908809 / +91-9911013303 "
        "or fill the enquiry form on our website.'\\n\\n"
        f"Context:\\n{context}\\n"
        f"Question: {q}\\n"
        "Answer in 2–4 short sentences."
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 400,
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return clean_answer(raw)
    except Exception as e:
        print("Groq error:", e)
        return None


# ------------ FLASK ROUTE ------------

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = ""
    mode_note = ""
    sources: List[str] = []

    if request.method == "POST":
        q = (request.form.get("query") or "").strip()

        if not q:
            answer = "Please type a question."
            return render_template_string(
                FORM_HTML,
                answer=answer,
                sources=[],
                mode_note=mode_note,
            )

        # 1 — Pricing first
        p = pricing_answer(q)
        if p:
            mode_note = "Pricing question — no AI used."
            answer = p
            return render_template_string(
                FORM_HTML,
                answer=answer,
                sources=[],
                mode_note=mode_note,
            )

        # 2 — FAQ
        f = faq_match(q)
        if f:
            mode_note = "FAQ matched."
            answer = (
                f
                + "\\n\\nFor a quick quote — call +91-11-42908809 / +91-9911013303 "
                  "or fill the enquiry form on our website."
            )
            answer = clean_answer(answer)
            sources = [f]
            return render_template_string(
                FORM_HTML,
                answer=answer,
                sources=sources,
                mode_note=mode_note,
            )

        # 3 — Website text
        retrieved = simple_retrieval(q)

        # 3a — Try Groq
        ai = call_groq_llm(q, retrieved)
        if ai:
            mode_note = "Groq AI used with your website content."
            answer = clean_answer(ai)
            sources = [s[:200] for s in retrieved]
        else:
            # 3b — No Groq fallback
            answer, srcs, mode_note = simple_answer(q, retrieved)
            sources = srcs

    return render_template_string(
        FORM_HTML,
        answer=answer,
        sources=sources,
        mode_note=mode_note,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
