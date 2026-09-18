from pathlib import Path
import torch


# ============================================================
# PROJECT PATH
# ============================================================

SEGFORMER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SEGFORMER_DIR.parent

# Dataset parquet untuk training
DATASET_ROOT = PROJECT_ROOT / "tcd_dataset" / "data"

TRAIN_PATTERN = "train-*.parquet"
TEST_PATTERN = "test-*.parquet"

# Dataset raster untuk prediction
TIF_DATASET_DIR = PROJECT_ROOT / "tcd_dataset" / "dataset"
TIF_PATTERN = "*.tif"


# ============================================================
# MODEL CHECKPOINT
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "segformer_b5"
)

BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

SEGFORMER_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segformer"
)

# Hasil semantic mask
SEMANTIC_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "semantic_masks"
)

# Hasil instance mask
INSTANCE_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "instance_masks"
)

# Semua overlay
OVERLAY_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "overlays"
)

# Overlay semantic
SEMANTIC_OVERLAY_DIR = (
    OVERLAY_OUTPUT_DIR
    / "semantic"
)

# Overlay bounding box
BBOX_OVERLAY_DIR = (
    OVERLAY_OUTPUT_DIR
    / "bbox"
)

# Preview detail
DETAIL_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "detail"
)

# Preview sederhana
PREVIEW_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "preview"
)

# Statistik
STATISTICS_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "statistics"
)

# Evaluasi atau metrics
METRICS_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "metrics"
)

# Combined frame
COMBINED_OUTPUT_DIR = (
    SEGFORMER_OUTPUT_DIR
    / "combined"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = (
    "nvidia/segformer-b5-finetuned-ade-640-640"
)

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
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

# Ukuran patch saat prediction
PREDICT_PATCH_SIZE = IMAGE_SIZE

# Overlap antar patch
PREDICT_OVERLAP = 128

# Batch patch saat prediction
PREDICT_BATCH_SIZE = 1


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1
NUM_EPOCHS = 2

LEARNING_RATE = 6e-5
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2
NUM_WORKERS = 0
SEED = 42


# ============================================================
# PREDICTION CONFIGURATION
# ============================================================

# True = semua file tif dalam TIF_DATASET_DIR
PREDICT_ALL_TIF = True

# Threshold probabilitas tree
TREE_THRESHOLD = 0.5

# Luas minimum objek instance
MIN_INSTANCE_AREA = 20

# Jarak minimum watershed
WATERSHED_MIN_DISTANCE = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    directories = [
        CHECKPOINT_DIR,
        BEST_MODEL_DIR,

        SEGFORMER_OUTPUT_DIR,

        SEMANTIC_OUTPUT_DIR,
        INSTANCE_OUTPUT_DIR,

        OVERLAY_OUTPUT_DIR,
        SEMANTIC_OVERLAY_DIR,
        BBOX_OVERLAY_DIR,

        DETAIL_OUTPUT_DIR,
        PREVIEW_OUTPUT_DIR,

        STATISTICS_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,

        COMBINED_OUTPUT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":
    create_directories()

    print("=" * 70)
    print("SEMUA DIREKTORI SEGFORMER BERHASIL DIBUAT")
    print("=" * 70)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Dataset raster     : {TIF_DATASET_DIR}")
    print(f"Checkpoint         : {BEST_MODEL_DIR}")
    print(f"Output utama       : {SEGFORMER_OUTPUT_DIR}")
    print(f"Combined output    : {COMBINED_OUTPUT_DIR}")
    print(f"Device             : {DEVICE}")