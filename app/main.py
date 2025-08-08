import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import *
from app.core.embeddings import OpenAIEmbeddingProvider
from app.dependencies.auth import auth

load_dotenv()

# Application state for storing loaded models
app_state = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Lifespan for the application
    """
    # Initialize embedding provider at startup
    app_state["embedding_provider"] = OpenAIEmbeddingProvider()
    yield
    # Clean up
    app_state.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL") or "http://localhost:3000"],  # Your frontend URL
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