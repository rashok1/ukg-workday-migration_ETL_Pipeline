import ollama

response = ollama.chat(
    model="llama3:latest",
    messages=[
        {
            "role": "user",
            "content": """
Normalize this HR department name.

Return ONLY JSON.

{
  "workday_value": "...",
  "confidence": 0.0,
  "reason": "..."
}

Input:
HR
"""
        }
    ]
)

print(response["message"]["content"])