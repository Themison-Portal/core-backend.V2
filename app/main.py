# Debug: confirm main.py is loading
print("MAIN.PY LOADED SUCCESSFULLY")
"""
Main application file
"""

import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.query import router as query_router
from app.api.routes.upload import router as upload_router
from app.dependencies.auth import auth

load_dotenv()

# Application state for storing loaded models
app_state = {}

app = FastAPI()

# CORS configuration for production
# Note: For production, specify exact origins instead of ['*'] for better security
allowed_origins = [
    "https://themison-mvp-v1.vercel.app",
    "http://localhost:8080",
    "http://localhost:5173",
]

# Allow all origins from environment variable if set
if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],  # Allow all methods including OPTIONS for preflight
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(
    auth_router,
    prefix="/auth",
    tags=["auth"]
)

# Public route for health check
@app.get("/")
def health():
    return {"status": "Service is running!"}

# Protected routes
app.include_router(
    upload_router,
    prefix="/upload",
    tags=["upload"],
    dependencies=[Depends(auth.verify_jwt)]
)

app.include_router(
    query_router,
    prefix="/query",
    tags=["query"],
    dependencies=[Depends(auth.verify_jwt)]
)