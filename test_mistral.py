import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

response = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "ministral-3b-2512",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    },
)

print("Status code:", response.status_code)
print("Response:", response.json())