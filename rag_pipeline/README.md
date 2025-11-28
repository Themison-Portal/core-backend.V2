Clinical Protocol RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for clinical trial protocols. It allows querying protocol PDFs, retrieves relevant chunks, generates answers with source-aware citations, and highlights referenced text in the PDFs.

The pipeline supports routine, emergency, and severe-side-effect queries, adjusting retrieval depth (top_k) based on query complexity.

Table of Contents

Project Structure

Setup

Data Placement

Database & Supabase

Redis Cache

Running the Pipeline

Querying via API

PDF Viewer & Highlighting

Citation Rules

Deployment Notes

Project Structure
project-root/
│
├─ data/                  # Place protocol PDFs here (e.g., obesity.pdf)
├─ main.py                # FastAPI server with query endpoint & PDF highlighting
├─ query_data_store.py    # RAG logic: embeddings, Supabase search, query classification
├─ models/
│   └─ vector_model.py    # SQLAlchemy models for Protocol and ProtocolChunk
├─ database.py            # Database connection and session setup
├─ static/                # Frontend templates & assets
│   └─ viewer.html        # PDF viewer with highlight support
└─ .env                   # Environment variables for OpenAI, Supabase, etc.

Setup

Clone the repository

git clone <company_repo_url>
cd <project-root>


Create a virtual environment & install dependencies

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt


Environment variables (.env)

OPENAI_API_KEY=<your_openai_key>
SUPABASE_DB_URL=<your_supabase_db_url>

Data Placement

Place all protocol PDFs in the data/ folder.

File names will be used as protocol titles (without .pdf extension).

Example:

data/
├─ obesity.pdf
├─ diabetes.pdf
└─ hypertension.pdf


During the data store generation, PDFs will be split into paragraph-level chunks and saved to Supabase.

Database & Supabase

Models:

Protocol: Stores each protocol PDF title.

ProtocolChunk: Stores paragraph-level chunks, embeddings, page, and paragraph numbers.

Vector embeddings: Generated using OpenAI text-embedding-3-small.

Database operations handled via SQLAlchemy.

Generate & save chunks:

python query_data_store.py

Redis Cache

Used for PDF block storage and fast text search.

Default connection: localhost:6379 (can be configured).

During startup, PDFs and JSON metadata are loaded into Redis for highlighting and bbox lookups.

Redis stores keys like:

blocks:<doc_stem>      # List of all text blocks with bbox info
highlighted:<cache_key> # Optional cached highlighted PDFs

Running the Pipeline

Start the FastAPI server:

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Home endpoint: http://localhost:8000/

Query form available at /

Querying via API

POST request to /query with form data:

POST /query
Content-Type: application/x-www-form-urlencoded

query=What is the recommended dosage for obese patients?


Response:

Main answer text (with inline citations)

SOURCES USED: section with all referenced PDFs

Query classification:

emergency / severe_side_effect → top_k=40

routine → top_k=15

PDF Viewer & Highlighting

Viewer endpoint: /viewer?doc=<pdf>&page=<num>&highlight=<text>

Highlights text in yellow on the specified page.

Multiple sentences or paragraph-level highlights supported.

Uses PyMuPDF for PDF manipulation.

Citation Rules

Inline citation format:

(p. PAGE_NUMBER, ¶PARAGRAPH_NUMBER)


Sources section at the end lists each referenced protocol once:

SOURCES USED:
Obesity (p. 40, ¶2)
Diabetes (p. 38, ¶1)


Do not fabricate page or paragraph numbers. Only use metadata from chunks.