import os
from dotenv import load_dotenv

try:
    load_dotenv(override=True, encoding='utf-8')
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

api_key = os.getenv("CHUTES_API_KEY")
if not api_key:
    raise ValueError("CHUTES_API_KEY not set in .env")
