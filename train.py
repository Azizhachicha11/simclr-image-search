import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_train_loader
from model import SimCLR


# ── Loss NT-Xent ──────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Pour un batch de N images :
      - On construit 2N embeddings [z1 ; z2]
      - La paire positive de z_i est z_{i+N} (et vice-versa)
      - Tous les autres sont des négatifs

    Args:
        z1, z2 : embeddings normalisés L2, shape (N, D)
        temperature : τ (typiquement 0.5)

    Returns:
        loss scalaire
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2N, D)

    # Matrice de similarité cosinus (2N x 2N)
    sim = torch.mm(z, z.T) / temperature      # (2N, 2N)

    # Masque : exclure la diagonale (similarité d'un vecteur avec lui-même)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Labels : la paire positive de i est i+N (et de i+N est i)
    labels = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(0, N)
    ]).to(z.device)                           # (2N,)

    loss = F.cross_entropy(sim, labels)
    return loss


# ── Boucle d'entraînement ─────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Données
    loader = get_train_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Modèle
    model = SimCLR(projection_dim=args.projection_dim).to(device)

    # Optimiseur : LARS est idéal pour SimCLR, mais Adam fonctionne bien aussi
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler cosinus (recommandé par le papier SimCLR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for view1, view2, _ in loop:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"  → Loss moyenne : {avg_loss:.4f} | LR : {scheduler.get_last_lr()[0]:.6f}")

        # Sauvegarde du meilleur modèle
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "simclr_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ Checkpoint sauvegardé → {ckpt_path}")

        # Sauvegarde périodique
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"simclr_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nEntraînement terminé. Meilleure loss : {best_loss:.4f}")


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement SimCLR sur STL-10")
    parser.add_argument("--data-root",       default="./data")
    parser.add_argument("--checkpoint-dir",  default="./checkpoints")
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--temperature",     type=float, default=0.5)
    parser.add_argument("--projection-dim",  type=int,   default=128)
    parser.add_argument("--num-workers",     type=int,   default=4)
    parser.add_argument("--save-every",      type=int,   default=10)
    args = parser.parse_args()
    train(args)