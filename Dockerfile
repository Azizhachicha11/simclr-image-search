# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — API SimCLR Image Search
# ══════════════════════════════════════════════════════════════════════════════
# Image légère Python 3.12 avec le backend FastAPI.
# Le modèle et l'index FAISS sont téléchargés depuis Hugging Face au démarrage.
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

# ── Variables d'environnement ─────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Répertoire de travail ─────────────────────────────────────────────────────
WORKDIR /app

# ── Installation des dépendances système ──────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Copie et installation des dépendances Python ─────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code backend ────────────────────────────────────────────────────
COPY backend/ .

# ── Copie du modèle et de l'index (si présents localement) ──────────────────
# Note : en production (Render), ceux-ci sont téléchargés depuis HF Hub
COPY checkpoints/ ./checkpoints/
COPY index/ ./index/

# ── Exposition du port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Lancement ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
