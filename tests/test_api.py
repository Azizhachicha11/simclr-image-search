"""
Tests d'intégration pour l'API FastAPI.

Couvre :
  - /health           → statut 200
  - /search (image)   → 5 résultats par défaut
  - /search (texte)   → erreur 400
  - /search?top_k=3   → exactement 3 résultats
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# ── Import de l'app FastAPI ───────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from api import app  # noqa: E402

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_test_image(width=96, height=96, color=(128, 64, 200)):
    """Crée une image RGB synthétique au format JPEG en mémoire."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health():
    """GET /health doit retourner 200 avec status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_search_valid_image():
    """POST /search avec une image valide doit retourner 5 résultats (top_k par défaut)."""
    image_bytes = _make_test_image()
    response = client.post(
        "/search",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 5
    # Chaque résultat doit avoir les champs attendus
    for result in data["results"]:
        assert "rank" in result
        assert "class" in result
        assert "similarity" in result


def test_search_invalid_file():
    """POST /search avec un fichier texte doit retourner 400."""
    fake_file = io.BytesIO(b"this is not an image")
    response = client.post(
        "/search",
        files={"image": ("fake.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


def test_search_top_k():
    """POST /search avec top_k=3 doit retourner exactement 3 résultats."""
    image_bytes = _make_test_image(color=(50, 100, 150))
    response = client.post(
        "/search",
        data={"top_k": "3"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
