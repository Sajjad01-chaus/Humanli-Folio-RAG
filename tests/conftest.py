import io
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen.canvas import Canvas

from app.config import Settings
from app.main import create_app


def pdf_bytes(*pages):
    stream = io.BytesIO()
    canvas = Canvas(stream)
    for text in pages:
        canvas.drawString(40, 780, text)
        canvas.showPage()
    canvas.save()
    return stream.getvalue()


@pytest.fixture
def settings(tmp_path):
    return replace(
        Settings(),
        data_dir=tmp_path,
        api_key="",
        provider="groq",
        embeddings="tfidf",
        rerank=False,
        adaptive=False,
        verify=False,
        ocr=False,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def uploaded(client):
    data = pdf_bytes(
        "Remote work is permitted three days per week.",
        "Customer support tickets are retained for 24 months after closure.",
    )
    response = client.post("/api/documents", files={"files": ("handbook.pdf", data, "application/pdf")})
    return response.json()["documents"][0]
