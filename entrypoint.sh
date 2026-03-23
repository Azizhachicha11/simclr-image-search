#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# entrypoint.sh — Télécharge le modèle depuis HuggingFace, puis lance l'API
# ══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════"
echo "  SimCLR Image Search — Starting up"
echo "═══════════════════════════════════════════════"

mkdir -p checkpoints index

echo "📥 Downloading model files from HuggingFace Hub..."
python -c "
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HF_TOKEN')
repo  = 'ElBOOH55/simclr-stl10'

print('  -> simclr_best.pt ...')
hf_hub_download(repo_id=repo, filename='simclr_best.pt', local_dir='checkpoints', token=token)

print('  -> faiss.index ...')
hf_hub_download(repo_id=repo, filename='faiss.index', local_dir='index', token=token)

print('  -> labels.npy ...')
hf_hub_download(repo_id=repo, filename='labels.npy', local_dir='index', token=token)

print('All files downloaded!')
"

echo "Starting FastAPI on port 8000..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
