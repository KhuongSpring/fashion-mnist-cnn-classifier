"""
Yêu cầu cài đặt thư viện (requirements):
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
pandas>=2.0
tqdm>=4.65
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION (Cấu hình)
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

# Tạo thư mục tự động nếu chưa có
for d in (DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

SEED = 42
# Fashion-MNIST global mean/std (single channel, pixel range [0,1])
MEAN, STD = (0.2860,), (0.3530,)

# ==========================================
# 2. UTILS (Tiện ích)
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_history(history: dict, title: str, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def report_metrics(y_true, y_pred, class_names, save_path=None):
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    print(report_str)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=2)
    return report_dict

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# ==========================================
# 3. DATA (Tiền xử lý và tải dữ liệu)
# ==========================================
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

# ==========================================
# 4. MODELS (Cấu trúc mạng)
# ==========================================
class MLP(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout / 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout / 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_model(name: str, dropout: float = 0.3, num_classes: int = 10):
    name = name.lower()
    if name == "mlp":
        return MLP(num_classes=num_classes, dropout=dropout)
    if name == "cnn":
        return CNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")

# ==========================================
# 5. ENGINE (Huấn luyện và đánh giá)
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

def fit(model, train_loader, val_loader, epochs, lr, weight_decay, device, verbose=True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    progress = tqdm(range(1, epochs + 1), desc="Epochs") if verbose else range(1, epochs + 1)
    
    for epoch in progress:
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

    return history

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

# ==========================================
# 6. MAIN (Hàm thực thi chính)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP/CNN on Fashion-MNIST")
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Chú ý: Nếu chạy trên Windows gặp lỗi BrokenPipe, hãy sửa num_workers=0 trong hàm get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, augment=not args.no_augment, num_workers=2
    )

    model = build_model(args.model, dropout=args.dropout)

    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    tag = args.model
    plot_history(history, title=tag.upper(), save_path=FIGURE_DIR / f"{tag}_history.png")
    with open(LOG_DIR / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    y_pred, y_true = predict(model, test_loader, device)
    report_metrics(y_true, y_pred, CLASS_NAMES, save_path=LOG_DIR / f"{tag}_report.json")
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES, title=f"{tag.upper()} Confusion Matrix",
        save_path=FIGURE_DIR / f"{tag}_confusion_matrix.png",
    )

    save_checkpoint(model, CHECKPOINT_DIR / f"{tag}.pt")
    print(f"Done. Checkpoint, figures and logs saved for '{tag}'.")

if __name__ == "__main__":
    main()