"""
Tests unitaires pour le modèle SimCLR.

Couvre :
  - Chargement du modèle sans erreur
  - Forme de sortie de encode() : (1, 512)
  - Normalisation L2 des vecteurs (norme ≈ 1.0)
"""

import torch
import pytest

import sys, os
# model.py is at the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import SimCLR  # noqa: E402

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "simclr_best.pt"
)


# ── Fixture : modèle chargé une seule fois pour tous les tests ────────────────
@pytest.fixture(scope="module")
def model():
    """Charge le modèle SimCLR depuis le checkpoint."""
    m = SimCLR(projection_dim=128)
    m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_image():
    """Crée un tenseur image factice (1, 3, 96, 96)."""
    return torch.randn(1, 3, 96, 96)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_model_loads(model):
    """Le modèle SimCLR se charge sans erreur."""
    assert model is not None
    assert isinstance(model, SimCLR)


def test_encode_shape(model, dummy_image):
    """encode() doit retourner un vecteur de forme (1, 512)."""
    embedding = model.encode(dummy_image)
    assert embedding.shape == (1, 512), f"Expected (1, 512), got {embedding.shape}"


def test_encode_normalized(model, dummy_image):
    """Les embeddings doivent être normalisés L2 (norme ≈ 1.0)."""
    embedding = model.encode(dummy_image)
    norm = torch.norm(embedding, p=2, dim=1).item()
    assert abs(norm - 1.0) < 1e-4, f"Norm should be ~1.0, got {norm}"
