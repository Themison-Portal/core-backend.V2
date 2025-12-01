#!/bin/bash
# start.sh

# Install dependencies
# pip install -r requirements.txt

# Set port (Vercel will override this)
PORT=${PORT:-8000}

# Start the FastAPI application with auto-reload
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload