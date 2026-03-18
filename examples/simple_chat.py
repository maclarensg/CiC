"""Simple chat example.

Run with:
    python examples/simple_chat.py
"""

from cic import CiCClient

client = CiCClient(model="sonnet")

messages = [
    {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
]

result = client.chat(messages)
print("Response:", result.content)
print("Model:", result.model)
print(f"Estimated tokens: {result.usage.total_tokens}")
