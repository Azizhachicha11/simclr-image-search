"""
FastAPI backend for SimCLR Image Search — Production version for Render.

Differences vs app.py (local Flask):
- No STL-10 dataset loaded (not available on Render, saves ~2.5 GB)
- Returns class + similarity score only (no base64 thumbnails)
- Runs with uvicorn on port 8000
- Loads model + FAISS index from ./checkpoints/ and ./index/
  (downloaded at startup by entrypoint.sh from Hugging Face Hub)
"""

import os
import io
import numpy as np
import torch
import faiss
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import SimCLR

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoints/simclr_best.pt")
INDEX_PATH      = os.environ.get("INDEX_PATH",      "./index/faiss.index")
LABELS_PATH     = os.environ.get("LABELS_PATH",     "./index/labels.npy")
DEFAULT_TOP_K   = 5

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SimCLR Image Search API",
    description="Visual similarity search using SimCLR embeddings + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load resources at startup ─────────────────────────────────────────────────
device = torch.device("cpu")  # Render Free = CPU only

print("Loading SimCLR model...")
model = SimCLR(projection_dim=128).to(device)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
)
model.eval()
print(f"Model loaded on {device}")

print("Loading FAISS index...")
faiss_index = faiss.read_index(INDEX_PATH)
labels = np.load(LABELS_PATH)
print(f"FAISS index ready: {faiss_index.ntotal} vectors")

# ── Image preprocessing ───────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def encode_image(image: Image.Image) -> np.ndarray:
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode(tensor)
    return embedding.cpu().numpy()  # (1, 512)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed": int(faiss_index.ntotal),
        "device": str(device),
    }


@app.post("/search")
async def search(
    image: UploadFile = File(...),
    top_k: int = Form(DEFAULT_TOP_K),
):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    top_k = max(1, min(top_k, 20))

    # Read + decode image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Encode + search
    query_emb = encode_image(img)
    distances, indices = faiss_index.search(query_emb, top_k)
    distances = distances[0]
    indices   = indices[0]

    # Build results
    results = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        label_idx = int(labels[int(idx)])
        results.append({
            "rank":       rank,
            "index":      int(idx),
            "class":      STL10_CLASSES[label_idx],
            "similarity": round(float(dist), 4),
        })

    return {
        "results":       results,
        "total_indexed": int(faiss_index.ntotal),
    }
