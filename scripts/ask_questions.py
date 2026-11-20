"""Ask the LLM our main questions with NO documents, and print its answer.

Run from the repo root:   python scripts/ask_questions.py

Standalone version: needs only 'requests' and 'python-dotenv' (already in requirements.txt).
"""
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Resolve project root so cache/qa_cache.json always points to <project>/cache/qa_cache.json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "cache" / "qa_cache.json"

# Load GROQ_API_KEY from the .env file in the project root
load_dotenv(PROJECT_ROOT / ".env")

# The common LLM imposed by the teacher for ALL teams. Never change it.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

# Terms indicating a question is outside the clinical/thesaurus sources
OUT_OF_SOURCE_TERMS = {
    "price",
    "prix",
    "cost",
    "coût",
    "cout",
    "tarif",
    "buy",
    "acheter",
    "stock",
    "discount",
    "promo",
    "weather",
    "météo",
    "meteo",
    "salary",
    "salaire",
}

QUESTIONS = {
    # Original question
    1: (
        "Un patient de 68 ans, hypertendu, est traité au long cours par du ramipril et de la "
        "spironolactone. Il se plaint de douleurs au genou depuis une semaine et demande de "
        "l'ibuprofène sans ordonnance à l'officine. Que disent les sources sur : "
        "(1) l'association de l'ibuprofène avec chacun de ces deux médicaments, et le niveau de "
        "chaque interaction ; (2) la dose maximale quotidienne d'ibuprofène chez l'adulte en vente "
        "libre ; (3) les précautions liées à l'âge du patient ; (4) quels autres antalgiques ou "
        "anti-inflammatoires de la liste ont des mises en garde avec ces deux traitements ? "
        "Cite la source de chaque information."
    ),
    # EN: What is the risk of combining ACIDE ACETYLSALICYLIQUE with ACETAZOLAMIDE?
    2: "Quel est le risque de l'association de l'ACIDE ACETYLSALICYLIQUE avec l'ACETAZOLAMIDE ?",
    # EN: How long after stopping ACITRETINE with ALCOOL does the teratogenic risk last?
    3: "Combien de temps après l'arrêt de l'ACITRETINE avec l'ALCOOL persiste le risque tératogène ?",
    # EN: What time gap is recommended between AFATINIB and AMIODARONE?
    4: "Quel écart horaire est recommandé entre l'AFATINIB et l'AMIODARONE ?",
    # EN: What is the max daily dose of SIMVASTATINE with AMLODIPINE?
    5: "Quelle est la dose quotidienne maximale de SIMVASTATINE avec l'AMLODIPINE ?",
    # EN: What is the max daily dose of ATORVASTATINE with CICLOSPORINE?
    6: "Quelle est la dose quotidienne maximale d'ATORVASTATINE avec la CICLOSPORINE ?",
    # EN: How many hours before iron salts should BICTÉGRAVIR be taken if not with a meal?
    7: "Combien d'heures avant les sels de fer le BICTÉGRAVIR doit-il être pris s'il n'est pas pris au cours d'un repas ?",
    # EN: For how many days after stopping ALLOPURINOL should ANTIVITAMINES K INR/dosage monitoring continue?
    8: "Pendant combien de jours après l'arrêt de l'ALLOPURINOL faut-il poursuivre la surveillance de l'INR / l'adaptation posologique des ANTIVITAMINES K ?",
    # EN: What is the average retail price of PARACETAMOL in local pharmacies?
    9: "Quel est le prix moyen de vente du PARACETAMOL en pharmacie ?",
}


def detect_out_of_source_term(question_text: str) -> str | None:
    """Returns the matched out-of-source term if present, otherwise None."""
    normalized = question_text.lower()
    words = set(re.findall(r"[a-zàâçéèêëîïôùûüÿñæœ]+", normalized))

    for term in sorted(OUT_OF_SOURCE_TERMS, key=len, reverse=True):
        if " " in term:
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                return term
        elif term in words:
            return term
    return None


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Old format (one question only, keys at the top level): keep it as question 1.
        if "answer" in data:
            data = {"1": {"question": QUESTIONS[1], **data}}
        return data
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def call_llm(question: str) -> dict:
    """Calls the common project LLM (openai/gpt-oss-120b) through the Groq API.

    Returns a dict: answer, finish_reason, total_tokens.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Check your .env file in the project root.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant pour pharmaciens d'officine. Réponds en français."},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_completion_tokens": 2500,   # the model "thinks" first; thinking uses part of this budget
        "reasoning_effort": "medium",
    }

    response = None
    for attempt in range(1, 4):          # at most 3 tries (rule for 429 errors)
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=90,
        )
        if response.status_code == 429:
            print(f"Rate limit (429). Waiting 20 seconds... (try {attempt}/3)")
            time.sleep(20)
            continue
        break

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:300]}")

    data = response.json()
    choice = data["choices"][0]
    return {
        "answer": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "total_tokens": data.get("usage", {}).get("total_tokens"),
    }


def generate_answer(question_text: str) -> dict:
    matched_term = detect_out_of_source_term(question_text)
    if matched_term is not None:
        return {
            "answer": (
                f"Question hors périmètre des sources (mot détecté : '{matched_term}'). "
                "Impossible de répondre à partir des sources fournies."
            ),
            "finish_reason": "blocked_by_keyword_filter",   # no API call was made
            "total_tokens": 0,
        }
    return call_llm(question_text)


def get_entry(q_id: int, cache: dict) -> tuple[dict, bool]:
    """Return (cache entry, made_real_api_call). Saves new answers to the cache file immediately."""
    cache_key = str(q_id)
    if cache_key in cache:
        return cache[cache_key], False

    question_text = QUESTIONS[q_id]
    result = generate_answer(question_text)
    entry = {
        "question": question_text,
        "answer": result["answer"],
        "finish_reason": result["finish_reason"],
        "total_tokens": result["total_tokens"],
    }
    cache[cache_key] = entry
    save_cache(cache)          # saved after EVERY question, so progress is never lost
    return entry, result["finish_reason"] != "blocked_by_keyword_filter"


def print_entry(q_id: int, entry: dict) -> None:
    print(f"\nQ{q_id}: {entry['question']}")
    print(f"Réponse:\n{entry['answer']}")
    print("-" * 60)
    print("Tokens used:", entry.get("total_tokens", "n/a"),
          "| finish_reason:", entry.get("finish_reason", "n/a"))
    if entry.get("finish_reason") == "length":
        print("WARNING: the answer was cut off (token limit reached). Raise max_completion_tokens.")


def run_all(cache: dict) -> None:
    """Ask every question that is not in the cache yet, pausing between real API calls."""
    pause_seconds = 20     # free plan: about 8,000 tokens/minute, so we space the calls out
    total = len(QUESTIONS)
    for q_id in QUESTIONS:
        already = str(q_id) in cache
        entry, made_call = get_entry(q_id, cache)
        status = "already in cache" if already else "NEW"
        print(f"Q{q_id}/{total} [{status}] tokens={entry.get('total_tokens', 'n/a')} "
              f"finish={entry.get('finish_reason', 'n/a')}")
        if made_call:
            time.sleep(pause_seconds)
    print(f"\nAll answers are stored in: {CACHE_FILE}")


def main():
    cache = load_cache()

    print("\nQuestions disponibles :")
    for q_id, text in QUESTIONS.items():
        flagged = detect_out_of_source_term(text)
        flag_note = f" [HORS SOURCE: '{flagged}']" if flagged else ""
        preview = text if len(text) <= 110 else text[:107] + "..."
        print(f"  {q_id}.{flag_note} {preview}")

    choice = input(f"\nChoisissez un numéro de question (1-{len(QUESTIONS)}) ou 'all' pour toutes : ").strip().lower()

    if choice == "all":
        run_all(cache)
        return

    if not choice.isdigit() or int(choice) not in QUESTIONS:
        print(f"Choix invalide. Entrez un nombre entre 1 et {len(QUESTIONS)}, ou 'all'.")
        return

    q_id = int(choice)
    entry, made_call = get_entry(q_id, cache)
    print("\n[APPEL LLM]" if made_call else "\n[CACHE HIT]")
    print_entry(q_id, entry)


if __name__ == "__main__":
    main()