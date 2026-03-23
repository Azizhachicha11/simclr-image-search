import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Télécharge STL-10 si pas déjà fait
dataset = torchvision.datasets.STL10(
    root="./data", split="train", download=True
)

classes = ["airplane", "bird", "car", "cat", "deer",
           "dog", "horse", "monkey", "ship", "truck"]

# Afficher une grille : 5 images par classe
fig, axes = plt.subplots(10, 5, figsize=(12, 22))
fig.suptitle("STL-10 Dataset — 5 exemples par classe", fontsize=16, fontweight="bold")

# Collecter 5 images par classe
from collections import defaultdict
class_images = defaultdict(list)
for img, label in dataset:
    if len(class_images[label]) < 5:
        class_images[label].append(img)
    if all(len(v) == 5 for v in class_images.values()):
        break

# Afficher
for row, (label_idx, images) in enumerate(sorted(class_images.items())):
    for col, img in enumerate(images):
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if col == 0:
            axes[row][col].set_ylabel(
                classes[label_idx], fontsize=12,
                fontweight="bold", rotation=0,
                labelpad=60, va="center"
            )

plt.tight_layout()
plt.savefig("stl10_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Grille sauvegardée → stl10_overview.png")