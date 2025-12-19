# baseline_chatbot.py
import ollama

def chatbot(user_message: str) -> str:
    """Simple chatbot - just responds to messages."""
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": 
             "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    return response['message']['content']

# Test it
print(chatbot("What's 25 * 17?"))
print(chatbot("What's the weather in Tokyo?"))
print(chatbot("Send an email to mail@ranjankumar.in"))