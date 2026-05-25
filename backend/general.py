from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from firebase_service import FirebaseService

try:
    load_dotenv(override=True, encoding='utf-8')
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

app = FastAPI(title="CCTV Requirements Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase
try:
    firebase = FirebaseService()
except FileNotFoundError as e:
    print(f"Warning: Firebase not configured: {e}")
    firebase = None

api_key = os.getenv("CHUTES_API_KEY")
if not api_key:
    raise ValueError("CHUTES_API_KEY not set in .env")
