from pathlib import Path
import torch


# ============================================================
# PROJECT PATH
# ============================================================

DEEPLAB_DIR = Path(__file__).resolve().parent
BASE_DIR = DEEPLAB_DIR.parent


# ============================================================
# DATASET
# ============================================================

DATASET_ROOT = BASE_DIR / "tcd_dataset" / "data"

TRAIN_PATTERN = "train-*.parquet"
TEST_PATTERN = "test-*.parquet"


# ============================================================
# MODEL TRAINING
# ============================================================

MODEL_NAME = "resnet50"

NUM_CLASSES = 2
IN_CHANNELS = 3

ENCODER_WEIGHTS = None


# ============================================================
# CHECKPOINT TRAINING
# ============================================================

BEST_MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "deeplabv3plus"
    / "checkpoints"
    / "best_deeplabv3plus.pth"
)


# ============================================================
# INPUT PREDICTION
# ============================================================

PREDICT_INPUT_DIR = (
    BASE_DIR
    / "tcd_dataset"
    / "dataset"
)


# ============================================================
# OUTPUT PREDICTION
# ============================================================

PREDICT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "deeplabv3plus"
    / "predictions"
)


# ============================================================
# SLIDING WINDOW
# ============================================================

PREDICT_TILE_SIZE = 512
PREDICT_OVERLAP = 64
PREDICT_BATCH_SIZE = 1


# ============================================================
# SEGMENTATION THRESHOLD
# ============================================================

TREE_THRESHOLD = 0.50


# ============================================================
# INSTANCE SEGMENTATION
# ============================================================

MIN_INSTANCE_AREA = 100
WATERSHED_MIN_DISTANCE = 20


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# Harus disamakan dengan preprocessing saat training.
# Jika training menggunakan normalisasi ImageNet, gunakan True.

NORMALIZE_INPUT = True

IMAGE_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGE_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LABEL
# ============================================================

ID2LABEL = {
    0: "background",
    1: "tree",
}

LABEL2ID = {
    "background": 0,
    "tree": 1,
}