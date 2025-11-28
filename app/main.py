"""
Main application file
"""

import os
import sys
# from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import router as auth_router
from app.api.routes.query import router as query_router
from app.api.routes.upload import router as upload_router
from app.dependencies.auth import auth

from rag_pipeline.router import router as rag_router
from rag_pipeline.helpers import STATIC_DIR, DATA_DIR
from contextlib import asynccontextmanager
import redis
import logging

# load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Application state for storing loaded models
app_state = {}

# --- Lifespan handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = None

    try:
        logging.info("Initializing Redis…")

        # --- 1) Connect to Redis ---
        try:
            redis_url = os.getenv("REDIS_URL")
            # redis_client = redis.Redis(host="localhost", port=6379, db=0)
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            app.state.redis_client = redis_client
            logging.info("Redis connection successful.")
        except Exception as e:
            logging.error(f"Redis connection failed: {e}")
            raise RuntimeError("Failed to connect to Redis") from e

        # --- 2) Preload PDF blocks ---
        try:
            from rag_pipeline.helpers import load_pdf_blocks_into_redis
            logging.info("Loading PDF blocks into Redis…")
            load_pdf_blocks_into_redis(redis_client)
            logging.info("PDF blocks loaded.")
        except Exception as e:
            logging.error(f"Error loading PDF blocks: {e}")
            # Continue running; depends if you want this to be fatal
            raise       

        
        yield

    finally:
        # --- 3) Shutdown cleanup ---
        if redis_client:
            try:
                redis_client.close()
                logging.info("Redis connection closed.")
            except Exception as e:
                logging.error(f"Error closing Redis connection: {e}")


app = FastAPI(lifespan=lifespan)
# app = FastAPI()

if os.path.isdir(STATIC_DIR):
    app.mount("/rag/static", StaticFiles(directory=STATIC_DIR), name="rag_static")

if os.path.isdir(DATA_DIR):
    app.mount("/rag/pdfs", StaticFiles(directory=DATA_DIR), name="rag_pdfs")

# CORS configuration for production
# Note: For production, specify exact origins instead of ['*'] for better security
allowed_origins = [
    "https://themison-mvp-v1.vercel.app",
    "https://core-frontendv2.vercel.app",
    "http://localhost:8080",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],  # Allow all methods including OPTIONS for preflight
    allow_headers=["*"],
    expose_headers=["*"],
)

# Allow all origins from environment variable if set
if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
    allowed_origins = ["*"]
logging.info(f"calling root endpoint with allowed origins")
@app.get("/")
def root():
    return {"status": "ok"}

app.include_router(
    rag_router,
    prefix="/rag",
    tags=["rag"]
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

