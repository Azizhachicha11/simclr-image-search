import os
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm

from dataset import get_index_loader
from model import SimCLR


# ── Extraction des embeddings ─────────────────────────────────────────────────
def extract_embeddings(model, loader, device):
    """
    Passe toutes les images du dataset dans l'encodeur
    et retourne les embeddings + labels.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extraction des embeddings"):
            images = images.to(device, non_blocking=True)
            embeddings = model.encode(images)          # (B, 512) normalisé L2
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels     = np.concatenate(all_labels,     axis=0)
    return embeddings, labels


# ── Construction de l'index FAISS ─────────────────────────────────────────────
def build_index(embeddings):
    """
    Construit un index FAISS à recherche exacte par produit intérieur.
    Puisque les embeddings sont normalisés L2, le produit intérieur
    équivaut à la similarité cosinus.
    """
    dim = embeddings.shape[1]                          # 512

    # IndexFlatIP : produit intérieur exact (Inner Product)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index FAISS construit : {index.ntotal} vecteurs de dimension {dim}")
    return index


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle entraîné
    model = SimCLR(projection_dim=128).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Modèle chargé depuis : {args.checkpoint}")

    # Charger les données
    loader = get_index_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Extraire les embeddings
    embeddings, labels = extract_embeddings(model, loader, device)
    print(f"Embeddings extraits : {embeddings.shape}")

    # Construire et sauvegarder l'index
    index = build_index(embeddings)
    os.makedirs(args.output_dir, exist_ok=True)

    index_path    = os.path.join(args.output_dir, "faiss.index")
    labels_path   = os.path.join(args.output_dir, "labels.npy")

    faiss.write_index(index, index_path)
    np.save(labels_path, labels)

    print(f"Index sauvegardé  → {index_path}")
    print(f"Labels sauvegardés → {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction de l'index FAISS")
    parser.add_argument("--checkpoint",   required=True, help="Chemin vers le .pt du modèle")
    parser.add_argument("--data-root",    default="./data")
    parser.add_argument("--output-dir",   default="./index")
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--num-workers",  type=int, default=4)
    args = parser.parse_args()
    main(args)