"""First connectivity test for the Groq API (shared project LLM)."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from the .env file
key = os.environ["GROQ_API_KEY"]

r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Reply with just: OK"}],
    },
    timeout=60,
)
print(r.status_code)
print(r.json()["choices"][0]["message"]["content"])