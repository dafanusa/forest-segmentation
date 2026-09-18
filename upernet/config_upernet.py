from pathlib import Path
import torch


# ============================================================
# PROJECT PATH
# ============================================================

UPERNET_DIR = Path(__file__).resolve().parent
BASE_DIR = UPERNET_DIR.parent

DATASET_ROOT = BASE_DIR / "tcd_dataset"
DATA_DIR = DATASET_ROOT / "data"


# ============================================================
# DATASET
# ============================================================

TRAIN_PATTERN = str(DATA_DIR / "train-*.parquet")
TEST_PATTERN = str(DATA_DIR / "test-*.parquet")


# ============================================================
# MODEL
# ============================================================

# Model UPerNet berbasis Swin Transformer Tiny
MODEL_NAME = "openmmlab/upernet-swin-tiny"

# Kelas segmentasi:
# 0 = background
# 1 = tree
NUM_CLASSES = 2

ID2LABEL = {
    0: "background",
    1: "tree",
}

LABEL2ID = {
    "background": 0,
    "tree": 1,
}


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 20
BATCH_SIZE = 2
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VAL_SIZE = 0.2
SEED = 42


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 512

IMAGE_COLUMN = "image"
MASK_COLUMN = "mask"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = BASE_DIR / "outputs" / "upernet"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREVIEW_DIR = OUTPUT_DIR / "previews"
LOG_DIR = OUTPUT_DIR / "logs"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_upernet.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_upernet.pth"

# ============================================================
# KONFIGURASI PREDIKSI UPERNET
# ============================================================

PREDICT_INPUT_DIR = (
    BASE_DIR / "tcd_dataset" / "dataset"
)

PREDICT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "upernet"
    / "predictions"
)

PREDICT_TILE_SIZE = 512
PREDICT_OVERLAP = 64
PREDICT_BATCH_SIZE = 1
TREE_THRESHOLD = 0.50
MIN_INSTANCE_AREA = 100
WATERSHED_MIN_DISTANCE = 20

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# EVALUATION
# ============================================================

# Folder ground-truth mask
#
# Contoh:
# tcd_dataset/
# ├── dataset/
# │   ├── 100.tif
# │   └── 108.tif
# │
# └── masks/
#     ├── 100_mask.tif
#     └── 108_mask.tif
#
EVAL_GT_DIR = (
    DATASET_ROOT / "masks"
)


# Folder hasil prediksi UPerNet
EVAL_PRED_DIR = (
    PREDICT_OUTPUT_DIR
)


# Folder output evaluasi
EVAL_OUTPUT_DIR = (
    OUTPUT_DIR / "evaluation"
)


# File hasil evaluasi
EVAL_METRICS_JSON = (
    EVAL_OUTPUT_DIR / "metrics_upernet.json"
)

EVAL_METRICS_CSV = (
    EVAL_OUTPUT_DIR / "metrics_upernet.csv"
)

EVAL_CONFUSION_MATRIX_CSV = (
    EVAL_OUTPUT_DIR / "confusion_matrix_upernet.csv"
)

EVAL_CONFUSION_MATRIX_PNG = (
    EVAL_OUTPUT_DIR / "confusion_matrix_upernet.png"
)


# Nama kelas
EVAL_CLASS_NAMES = [
    "background",
    "tree",
]

# ============================================================
# EARLY STOPPING
# ============================================================

PATIENCE = 3
MIN_DELTA = 1e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda"