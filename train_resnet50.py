from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.resnet50 import resnet50


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "data"
    / "input"
    / "Dan_2014_RGB_project_to_CHM.tif"
)

MASK_PATH = (
    BASE_DIR
    / "outputs"
    / "Danum_resnet50_mask.tif"
)

BACKBONE_CHECKPOINT = (
    BASE_DIR
    / "checkpoints"
    / "resnet50s-a75c83cf.pth"
)

OUTPUT_CHECKPOINT = (
    BASE_DIR
    / "checkpoints"
    / "resnet50_segmentation_danum.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

TILE_SIZE = 256

BATCH_SIZE = 4

EPOCHS = 20

LEARNING_RATE_BACKBONE = 1e-5

LEARNING_RATE_HEAD = 1e-4

NUM_WORKERS = 0


# ============================================================
# MODEL
# ============================================================

class ResNet50Segmentation(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = resnet50(
            pretrained=False,
            num_classes=1000
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                2048,
                512,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        backbone_output = self.backbone(x)

        if isinstance(backbone_output, (tuple, list)):
            deep_feature = backbone_output[-1]
        else:
            deep_feature = backbone_output

        output = self.classifier(deep_feature)

        output = F.interpolate(
            output,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return output


# ============================================================
# LOAD BACKBONE CHECKPOINT
# ============================================================

def load_backbone_checkpoint(model, checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Hanya masukkan bobot backbone.
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]

        # Abaikan FC klasifikasi jika ada.
        if new_key.startswith("fc."):
            continue

        if new_key.startswith("classifier."):
            continue

        cleaned_state_dict[new_key] = value

    missing, unexpected = model.backbone.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("\n========== LOAD BACKBONE ==========")
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if missing:
        print("Contoh missing:")
        for key in missing[:10]:
            print(f"  - {key}")

    print("===================================\n")


# ============================================================
# NORMALISASI
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(3, 1, 1)

    image = (image - mean) / std

    return image


# ============================================================
# DATASET TILE
# ============================================================

class RasterSegmentationDataset(Dataset):

    def __init__(
        self,
        image_path,
        mask_path,
        tile_size=256
    ):

        self.image_path = image_path
        self.mask_path = mask_path
        self.tile_size = tile_size

        with rasterio.open(image_path) as src:

            self.width = src.width
            self.height = src.height

            if src.count < 3:
                raise ValueError(
                    "Citra harus memiliki minimal 3 band."
                )

        with rasterio.open(mask_path) as mask_src:

            if mask_src.width != self.width:
                raise ValueError(
                    "Lebar citra dan mask tidak sama."
                )

            if mask_src.height != self.height:
                raise ValueError(
                    "Tinggi citra dan mask tidak sama."
                )

        self.positions = []

        for top in range(
            0,
            self.height - tile_size + 1,
            tile_size
        ):

            for left in range(
                0,
                self.width - tile_size + 1,
                tile_size
            ):

                self.positions.append(
                    (top, left)
                )

        print(
            f"[INFO] Jumlah tile training: "
            f"{len(self.positions)}"
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):

        top, left = self.positions[index]

        window = rasterio.windows.Window(
            col_off=left,
            row_off=top,
            width=self.tile_size,
            height=self.tile_size
        )

        with rasterio.open(self.image_path) as src:

            image = src.read(
                indexes=[1, 2, 3],
                window=window
            )

        with rasterio.open(self.mask_path) as src:

            mask = src.read(
                1,
                window=window
            )

        image = normalize_image(image)

        image = torch.from_numpy(
            image
        ).float()

        # Pastikan label hanya 0 dan 1.
        mask = np.where(mask > 0, 1, 0)

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


# ============================================================
# LOSS
# ============================================================

def segmentation_loss(logits, target):

    ce_loss = nn.CrossEntropyLoss()

    return ce_loss(logits, target)


# ============================================================
# TRAINING
# ============================================================

def train():

    print("=" * 65)
    print("TRAINING RESNET50 SEGMENTATION")
    print("=" * 65)

    print(f"Device : {DEVICE}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Citra tidak ditemukan:\n{IMAGE_PATH}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"Mask ground truth tidak ditemukan:\n{MASK_PATH}"
        )

    dataset = RasterSegmentationDataset(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        tile_size=TILE_SIZE
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset tile kosong."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    model = ResNet50Segmentation(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    load_backbone_checkpoint(
        model,
        BACKBONE_CHECKPOINT
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": LEARNING_RATE_BACKBONE
            },
            {
                "params": model.classifier.parameters(),
                "lr": LEARNING_RATE_HEAD
            }
        ],
        weight_decay=1e-4
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_index, (images, masks) in enumerate(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = segmentation_loss(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 10 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        average_loss = total_loss / len(loader)

        print(
            f"\n[INFO] Epoch {epoch + 1} selesai | "
            f"Average Loss: {average_loss:.6f}\n"
        )

        if average_loss < best_loss:

            best_loss = average_loss

            OUTPUT_CHECKPOINT.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "tile_size": TILE_SIZE,
                    "best_loss": best_loss
                },
                OUTPUT_CHECKPOINT
            )

            print(
                f"[INFO] Checkpoint terbaik disimpan:\n"
                f"{OUTPUT_CHECKPOINT}"
            )

    print("\nTraining selesai.")


if __name__ == "__main__":
    train()