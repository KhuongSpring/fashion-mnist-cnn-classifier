import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import DATA_DIR, SEED

# Fashion-MNIST global mean/std (single channel, pixel range [0,1])
MEAN, STD = (0.2860,), (0.3530,)


def build_transforms(augment: bool):
    train_ops = [transforms.ToTensor()]
    if augment:
        train_ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    train_ops.append(transforms.Normalize(MEAN, STD))

    eval_ops = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_dataloaders(batch_size=128, augment=True, val_split=0.1, num_workers=2):
    train_tf, eval_tf = build_transforms(augment)

    full_train = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=train_tf
    )
    # Separate copy with eval transform, so the validation split is not augmented.
    full_train_eval = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=eval_tf
    )
    test_set = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=eval_tf
    )

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(SEED)

    train_indices, val_indices = random_split(
        range(len(full_train)), [n_train, n_val], generator=generator
    )

    train_set = torch.utils.data.Subset(full_train, train_indices.indices)
    val_set = torch.utils.data.Subset(full_train_eval, val_indices.indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
