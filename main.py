import argparse
import json

from src.config import CHECKPOINT_DIR, CLASS_NAMES, FIGURE_DIR, LOG_DIR, SEED
from src.data import get_dataloaders
from src.engine import fit, predict
from src.models import build_model
from src.utils import get_device, plot_confusion_matrix, plot_history, report_metrics, save_checkpoint, set_seed


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

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, augment=not args.no_augment
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
