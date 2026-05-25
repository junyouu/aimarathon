from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_service import FirebaseService

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
