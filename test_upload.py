import os
from fastapi.testclient import TestClient
from app.main import app  # import your FastAPI app
from app.dependencies.auth import get_current_user
from uuid import UUID

def real_user_override():
    return {
        "id": UUID("49ccce71-6eec-4414-b7c0-0b547528c110"),  # REAL user id
        "email": "iftikhar@themison.com"
    }

app.dependency_overrides[get_current_user] = real_user_override

client = TestClient(app)

def test_upload_pdf_document():
    payload = {
        "document_url": "https://gpfyejxokywdkudkeywv.supabase.co/storage/v1/object/public/trial-documents/1c8bab48-aed9-471d-bb49-030170ae589d/1764149955812-protocol_oncology.pdf",
        "document_id": "28ca8c51-056d-4296-8881-90bd0ce41653",
        "chunk_size": 750
    }

    response = client.post(
        "/upload/upload-pdf",
        json=payload
    )

    print(response.status_code)
    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_upload_pdf_document()