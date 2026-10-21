import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing in .env file")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents="Explain RAG in simple English in 3 lines."
)

print(response.text)