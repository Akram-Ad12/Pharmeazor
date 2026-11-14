"""Ask the LLM our main question (Q1) with NO documents, and print its answer.

Run from the repo root:   python scripts/ask_q1.py

Standalone version: needs only 'requests' and 'python-dotenv' (already in requirements.txt).
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]      # the Pharmeazor folder
load_dotenv(ROOT / ".env")                      # puts GROQ_API_KEY into the environment
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Check that your .env file is in the Pharmeazor folder.")

# --- 2. The question (in French, like a real pharmacist's question) --------------------
QUESTION = (
    "Un patient de 68 ans, hypertendu, est traité au long cours par du ramipril et de la "
    "spironolactone. Il se plaint de douleurs au genou depuis une semaine et demande de "
    "l'ibuprofène sans ordonnance à l'officine. Que disent les sources sur : "
    "(1) l'association de l'ibuprofène avec chacun de ces deux médicaments, et le niveau de "
    "chaque interaction ; (2) la dose maximale quotidienne d'ibuprofène chez l'adulte en vente "
    "libre ; (3) les précautions liées à l'âge du patient ; (4) quels autres antalgiques ou "
    "anti-inflammatoires de la liste ont des mises en garde avec ces deux traitements ? "
    "Cite la source de chaque information."
)

# --- 3. If already asked, reuse the saved answer (saves our daily quota) --------------
CACHE_FILE = ROOT / "cache" / "q1_answer.json"
if CACHE_FILE.exists():
    saved = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    from_cache = True
else:
    # --- 4. Send the question to Groq ---------------------------
    payload = {
        "model": "openai/gpt-oss-120b",       
        "messages": [
            {"role": "system", "content": "Tu es un assistant pour pharmaciens d'officine. Réponds en français."},
            {"role": "user", "content": QUESTION},
        ],
        "temperature": 0,                        
        "max_completion_tokens": 2500,            
        "reasoning_effort": "medium",
    }
    response = None
    for attempt in range(1, 4):                   # at most 3 tries 
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=90,
        )
        if response.status_code == 429:           # too many requests: wait, then retry
            print(f"Rate limit (429). Waiting 20 seconds... (try {attempt}/3)")
            time.sleep(20)
            continue
        break
    print("HTTP status:", response.status_code, "(200 = success)")
    if response.status_code != 200:
        raise SystemExit(f"Failed: {response.text[:300]}")

    data = response.json()
    saved = {
        "answer": data["choices"][0]["message"].get("content") or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "total_tokens": data.get("usage", {}).get("total_tokens"),
    }
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    from_cache = False

# --- 5. Show the answer ------------------------------------------------------------------
print("=" * 70)
print("ANSWER FROM THE LLM (no documents, no sources given to it):")
print("=" * 70)
print(saved["answer"])
print("=" * 70)
print("Loaded from cache (no API call used):", from_cache)
print("Tokens used:", saved["total_tokens"], "| finish_reason:", saved["finish_reason"])
if saved["finish_reason"] == "length":
    print("WARNING: the answer was cut off. Tell me and we will raise max_completion_tokens.")