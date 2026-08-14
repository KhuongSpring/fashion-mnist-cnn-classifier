from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

for d in (DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

SEED = 42

DEFAULT_HPARAMS = {
    "batch_size": 128,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "val_split": 0.1,
    "augment": True,
}
