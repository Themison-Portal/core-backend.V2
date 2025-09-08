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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL") or "http://localhost:3000"],  # Your frontend URL
    "http://localhost:3000", 
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(
    auth_router,
    prefix="/auth",
    tags=["auth"]
)

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