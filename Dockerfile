# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — API SimCLR Image Search (Production — Render)
# ══════════════════════════════════════════════════════════════════════════════
# Structure :
#   backend/api.py       → FastAPI app (production, pas le Flask local)
#   model.py             → SimCLR model (racine du projet)
#   backend/requirements.txt → dépendances
#   entrypoint.sh        → télécharge modèle depuis HF Hub puis lance uvicorn
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

# ── Code source ───────────────────────────────────────────────────────────────
# model.py est à la racine, api.py est dans backend/
COPY model.py .
COPY backend/api.py .

# ── Script de démarrage ───────────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./entrypoint.sh"]
