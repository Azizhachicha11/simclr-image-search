import argparse
import numpy as np
import torch
import faiss
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # pas besoin d'écran
import matplotlib.pyplot as plt
import torchvision

from model import SimCLR

# Noms des classes STL-10
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

# ── Prétraitement de l'image requête ─────────────────────────────────────────
def preprocess(image_path):
    transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, 3, 96, 96)


# ── Dénormalisation pour affichage ───────────────────────────────────────────
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ── Recherche ─────────────────────────────────────────────────────────────────
def search(query_embedding, index, top_k):
    """
    Recherche les top_k voisins les plus proches dans l'index FAISS.
    Retourne les distances et indices.
    """
    D, I = index.search(query_embedding, top_k)
    return D[0], I[0]


# ── Affichage des résultats ───────────────────────────────────────────────────
def display_results(query_path, indices, distances, labels, dataset, output_path):
    """
    Sauvegarde une grille : image requête + top-k résultats.
    """
    n = len(indices)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

    # Image requête
    query_img = Image.open(query_path).convert("RGB").resize((96, 96))
    axes[0].imshow(query_img)
    axes[0].set_title("REQUÊTE", fontsize=9, fontweight="bold", color="red")
    axes[0].axis("off")

    # Résultats
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        img_tensor, label = dataset[idx]
        img = denormalize(img_tensor).permute(1, 2, 0).numpy()
        axes[rank].imshow(img)
        axes[rank].set_title(
            f"#{rank} {STL10_CLASSES[label]}\nsim={dist:.3f}",
            fontsize=8
        )
        axes[rank].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Résultats sauvegardés → {output_path}")


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = SimCLR(projection_dim=128).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Modèle chargé depuis : {args.checkpoint}")

    # Charger l'index FAISS et les labels
    index  = faiss.read_index(args.index)
    labels = np.load(args.labels)
    print(f"Index chargé : {index.ntotal} images indexées")

    # Encoder l'image requête
    query_tensor = preprocess(args.query).to(device)
    query_emb    = model.encode(query_tensor).cpu().numpy()   # (1, 512)

    # Recherche
    distances, indices = search(query_emb, index, top_k=args.top_k)

    # Affichage texte
    print(f"\nTop-{args.top_k} résultats pour : {args.query}")
    print("-" * 40)
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        print(f"  #{rank}  idx={idx:5d}  classe={STL10_CLASSES[labels[idx]]:<10s}  similarité={dist:.4f}")

    # Affichage visuel
    if args.output:
        dataset = torchvision.datasets.STL10(
            root=args.data_root, split="train", download=True,
            transform=T.Compose([
                T.Resize((96, 96)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        )
        display_results(args.query, indices, distances, labels, dataset, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moteur de recherche d'images par similarité")
    parser.add_argument("--query",       required=True,  help="Chemin vers l'image requête")
    parser.add_argument("--checkpoint",  required=True,  help="Chemin vers simclr_best.pt")
    parser.add_argument("--index",       default="./index/faiss.index")
    parser.add_argument("--labels",      default="./index/labels.npy")
    parser.add_argument("--data-root",   default="./data")
    parser.add_argument("--top-k",       type=int, default=5)
    parser.add_argument("--output",      default="results.png", help="Image de sortie avec la grille")
    args = parser.parse_args()
    main(args)