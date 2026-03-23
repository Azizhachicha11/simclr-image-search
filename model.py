import torch
import torch.nn as nn
import torchvision.models as models


# ── Tête de projection ────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    MLP à 2 couches appliqué sur les embeddings de l'encodeur.
    SimCLR montre que la loss est calculée sur z (après projection),
    mais les embeddings utilisés pour la recherche sont h (avant projection).
    """
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Modèle SimCLR complet ─────────────────────────────────────────────────────
class SimCLR(nn.Module):
    """
    Encodeur ResNet18 + tête de projection.

    forward() retourne :
      - h : embedding avant projection  (utilisé pour FAISS)
      - z : embedding après projection  (utilisé pour la loss NT-Xent)
    """
    def __init__(self, projection_dim=128):
        super().__init__()

        # Encodeur : ResNet18 sans la couche de classification finale
        backbone = models.resnet18(weights=None)
        embedding_dim = backbone.fc.in_features          # 512 pour ResNet18
        backbone.fc = nn.Identity()                      # supprime le classifieur

        self.encoder = backbone
        self.projector = ProjectionHead(
            in_dim=embedding_dim,
            hidden_dim=embedding_dim,
            out_dim=projection_dim,
        )

    def forward(self, x):
        h = self.encoder(x)          # (B, 512)  ← embedding "utile"
        z = self.projector(h)        # (B, 128)  ← embedding pour la loss
        z = nn.functional.normalize(z, dim=1)   # normalisation L2
        return h, z

    def encode(self, x):
        """Retourne uniquement h (pour indexation / recherche)."""
        with torch.no_grad():
            h = self.encoder(x)
        return nn.functional.normalize(h, dim=1)