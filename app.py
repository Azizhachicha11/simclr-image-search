"""
Interactive Web Interface for SimCLR Image Search Engine.
Flask server that loads the trained model + FAISS index and serves
a drag-and-drop image search UI.
"""

import os
import io
import base64
import numpy as np
import torch
import faiss
import torchvision
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request, jsonify

from model import SimCLR

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints/simclr_best.pt"
INDEX_PATH      = "./index/faiss.index"
LABELS_PATH     = "./index/labels.npy"
DATA_ROOT       = "./data"
DEFAULT_TOP_K   = 5

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global resources (loaded once at startup) ─────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimCLR(projection_dim=128).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()
print(f"✔ Model loaded from {CHECKPOINT_PATH} on {device}")

# Load FAISS index & labels
faiss_index = faiss.read_index(INDEX_PATH)
labels = np.load(LABELS_PATH)
print(f"✔ FAISS index loaded: {faiss_index.ntotal} vectors")

# Load STL-10 dataset (for retrieving result images)
stl10_dataset = torchvision.datasets.STL10(
    root=DATA_ROOT, split="train", download=False, transform=None,
)
print(f"✔ STL-10 dataset loaded: {len(stl10_dataset)} images")

# Preprocessing transform (matches search.py)
preprocess_transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Helper Functions ──────────────────────────────────────────────────────────
def pil_to_base64(img, fmt="PNG", max_size=192):
    """Convert a PIL image to a base64-encoded string for embedding in HTML."""
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_query(image: Image.Image):
    """Preprocess and encode a query image to a 512-dim embedding."""
    tensor = preprocess_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode(tensor)
    return embedding.cpu().numpy()  # (1, 512)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_route():
    """Handle image upload, encode, search FAISS, return results as JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    top_k = request.form.get("top_k", DEFAULT_TOP_K, type=int)
    top_k = max(1, min(top_k, 20))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Encode query
    query_emb = encode_query(img)

    # FAISS search
    distances, indices = faiss_index.search(query_emb, top_k)
    distances = distances[0]
    indices = indices[0]

    # Build results with base64 thumbnails
    results = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        idx = int(idx)
        label_idx = int(labels[idx])
        result_img, _ = stl10_dataset[idx]  # PIL image
        results.append({
            "rank": rank,
            "index": idx,
            "class": STL10_CLASSES[label_idx],
            "similarity": round(float(dist), 4),
            "thumbnail": pil_to_base64(result_img),
        })

    # Also return query image thumbnail
    query_thumb = pil_to_base64(img)

    return jsonify({
        "query_thumbnail": query_thumb,
        "results": results,
        "total_indexed": int(faiss_index.ntotal),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
