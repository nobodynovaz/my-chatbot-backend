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
  <title>Live Broadcasting India — Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">

  <style>
    * {
      font-family: 'Poppins', sans-serif;
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      background: #E7EEF9;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }

    .chat-container {
      width: 100%;
      max-width: 850px;
      background: #fff;
      border-radius: 20px;
      box-shadow: 0 8px 26px rgba(0,0,0,0.1);
      overflow: hidden;
      border: 1px solid #d9e5ff;
    }

    .chat-header {
      background: linear-gradient(to right, #3F51E0, #4A58E8);
      padding: 20px 28px;
      color: white;
      font-size: 1.15rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .chat-header span {
      opacity: 0.9;
      font-size: 0.85rem;
      font-weight: 400;
    }

    .chat-body {
      padding: 24px 28px;
      max-height: 450px;
      overflow-y: auto;
    }

    .empty-message {
      text-align: center;
      font-size: 0.95rem;
      padding: 20px 0;
      color: #6b7280;
    }

    .mode-note {
      font-size: 0.85rem;
      color: #4A58E8;
      margin-bottom: 10px;
      opacity: 0.8;
    }

    .response {
      font-size: 1rem;
      line-height: 1.6;
      color: #222;
      white-space: pre-line;
      margin-bottom: 14px;
    }

    .sources-title {
      margin-top: 10px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      color: #4A58E8;
      opacity: 0.8;
    }

    .sources-list {
      padding: 0;
      margin-top: 6px;
      margin-bottom: 4px;
      list-style: inside disc;
      font-size: 0.85rem;
      color: #555;
      opacity: 0.9;
    }

    .chat-footer {
      padding: 18px 28px;
      background: #F5F7FF;
      border-top: 1px solid #dce4fc;
    }

    form {
      display: flex;
      gap: 12px;
    }

    input[name="query"] {
      flex: 1;
      border-radius: 30px;
      border: 1px solid #c5d4ff;
      padding: 12px 18px;
      font-size: 0.95rem;
      outline: none;
      transition: 0.2s;
    }

    input[name="query"]::placeholder {
      color: #9ca3af;
    }

    input[name="query"]:focus {
      border-color: #4A58E8;
      box-shadow: 0 0 0 2px rgba(74, 88, 232, 0.25);
    }

    button {
      background: #4A58E8;
      border: none;
      color: white;
      padding: 12px 22px;
      border-radius: 30px;
      font-weight: 500;
      cursor: pointer;
      transition: 0.2s;
    }

    button:hover {
      background: #343ECC;
    }

    button:active {
      transform: translateY(1px);
    }

    @media(max-width:700px) {
      form {
        flex-direction: column;
      }
      button {
        width: 100%;
      }
    }

  </style>
</head>

<body>
  <div class="chat-container">
    <div class="chat-header">
      Live Broadcasting Assistant
      <span>Powered by AI + Your Website</span>
    </div>

    <div class="chat-body">
      {% if answer %}

        {% if mode_note %}
          <div class="mode-note">{{ mode_note }}</div>
        {% endif %}

        <div class="response">
          {{ answer|replace('\\n', '<br>')|safe }}
        </div>

        {% if sources %}
          <div class="sources-title">Information used from website:</div>
          <ul class="sources-list">
            {% for s in sources %}
              <li>{{ s }}</li>
            {% endfor %}
          </ul>
        {% endif %}

      {% else %}
        <div class="empty-message">
          Ask anything like:<br><br>
          • "What services do you offer?"<br>
          • "Do you stream weddings?"<br>
          • "How much for a 5-camera cricket setup?"<br>
        </div>
      {% endif %}
    </div>

    <div class="chat-footer">
      <form method="post">
        <input name="query" placeholder="Ask about live streaming, pricing, services…" required>
        <button type="submit">Ask Now</button>
      </form>
    </div>
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
