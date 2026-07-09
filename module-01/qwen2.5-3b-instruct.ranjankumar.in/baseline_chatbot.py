# baseline_chatbot.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QWEN_URL = os.getenv("QWEN_URL") 
QWEN_API_KEY = os.getenv("QWEN_API_KEY")


def chatbot(user_message: str) -> str:
    """
    Simple chatbot client for Qwen2.5-3B-Instruct
    (OpenAI-compatible API)
    """
    payload = {
        "model": "qwen2.5-3b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        QWEN_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ---- Test it ----
if __name__ == "__main__":
    print(chatbot("What's 25 * 17?"))
    print(chatbot("What's the weather in Tokyo?"))
    print(chatbot("Send an email to mail@ranjankumar.in"))