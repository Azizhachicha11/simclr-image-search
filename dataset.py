import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# ── Augmentations SimCLR ──────────────────────────────────────────────────────
class SimCLRAugmentation:
    """
    Applique deux augmentations aléatoires différentes à la même image.
    Retourne une paire (vue1, vue2) utilisée comme paire positive.
    """
    def __init__(self, image_size=96):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9)], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class STL10Contrastive(Dataset):
    """
    Wrapper autour de STL-10 qui retourne des paires (vue1, vue2)
    pour l'entraînement contrastif.
    """
    def __init__(self, root="./data", split="unlabeled", download=True):
        self.augment = SimCLRAugmentation(image_size=96)
        self.dataset = torchvision.datasets.STL10(
            root=root, split=split, download=download,
            transform=None  # on gère la transform nous-mêmes
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        view1, view2 = self.augment(image)
        return view1, view2, label


# ── Dataset pour l'indexation (sans augmentation) ────────────────────────────
def get_index_dataset(root="./data", split="train"):
    """
    Dataset STL-10 avec transform simple pour construire l'index FAISS.
    Utilise le split 'train' (avec labels) pour pouvoir évaluer visuellement.
    """
    transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return torchvision.datasets.STL10(
        root=root, split=split, download=True, transform=transform
    )


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_train_loader(root="./data", batch_size=256, num_workers=0):
    dataset = STL10Contrastive(root=root, split="unlabeled", download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,   # important : NT-Xent suppose des batchs complets
    )


def get_index_loader(root="./data", batch_size=128, num_workers=0):
    dataset = get_index_dataset(root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
