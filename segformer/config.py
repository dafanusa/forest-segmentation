from pathlib import Path
import torch


# ============================================================
# ROOT PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET PARQUET
# ============================================================

DATASET_ROOT = PROJECT_ROOT / "tcd_dataset" / "data"

TRAIN_PATTERN = "train-*.parquet"
TEST_PATTERN = "test-*.parquet"


# ============================================================
# CITRA TIF UNTUK INFERENCE
# ============================================================

TIF_IMAGE_PATH = (
    PROJECT_ROOT
    / "tcd_dataset"
    / "dataset"
    / "713.tif"
)


# ============================================================
# OUTPUT
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "segformer_b5"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEMANTIC_OUTPUT_DIR = OUTPUT_DIR / "semantic_masks"
INSTANCE_OUTPUT_DIR = OUTPUT_DIR / "instance_masks"
OVERLAY_OUTPUT_DIR = OUTPUT_DIR / "overlays"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

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

IMAGE_SIZE = 512

BATCH_SIZE = 1

NUM_EPOCHS = 2

LEARNING_RATE = 6e-5

WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2

NUM_WORKERS = 0

SEED = 42


# ============================================================
# INSTANCE SEGMENTATION
# ============================================================

TREE_THRESHOLD = 0.5

MIN_INSTANCE_AREA = 20

WATERSHED_MIN_DISTANCE = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_directories():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEMANTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INSTANCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OVERLAY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )