# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — API SimCLR Image Search
# ══════════════════════════════════════════════════════════════════════════════
# Structure du projet : les fichiers Python sont à la RACINE (api.py, model.py...)
# Le modèle et l'index FAISS sont téléchargés depuis HuggingFace au démarrage.
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dépendances système ───────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source (racine du projet = api.py, model.py, search.py...) ──────────
COPY api.py model.py search.py dataset.py train.py index.py ./

# ── Script de démarrage ───────────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./entrypoint.sh"]
