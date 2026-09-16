from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import rasterio
import torch
import torchvision

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

OUTPUT_DIR = BASE_DIR / "outputs_single_713"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2
# 0 = Background
# 1 = Tree

NUM_EPOCHS = 20
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

BATCH_SIZE = 1
NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class TreeInstanceDataset(Dataset):

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_paths = sorted(
            image_dir.glob("*.png")
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"Tidak ada citra PNG di {image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ):

        image_path = self.image_paths[index]

        mask_path = (
            self.mask_dir
            / f"{image_path.stem}.tif"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(
            image
        )

        with rasterio.open(mask_path) as src:
            instance_mask = src.read(1)

        instance_mask = instance_mask.astype(
            np.int32
        )

        object_ids = np.unique(
            instance_mask
        )

        object_ids = object_ids[
            object_ids != 0
        ]

        masks = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for object_id in object_ids:

            object_mask = (
                instance_mask == object_id
            )

            rows, cols = np.where(
                object_mask
            )

            if len(rows) == 0:
                continue

            xmin = cols.min()
            ymin = rows.min()
            xmax = cols.max()
            ymax = rows.max()

            # Hindari bounding box dengan ukuran nol.
            if xmax <= xmin or ymax <= ymin:
                continue

            masks.append(
                object_mask
            )

            boxes.append(
                [
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ]
            )

            labels.append(1)

            areas.append(
                float(np.sum(object_mask))
            )

            iscrowd.append(0)

        if len(masks) == 0:
            masks_tensor = torch.zeros(
                (
                    0,
                    image.shape[0],
                    image.shape[1],
                ),
                dtype=torch.uint8,
            )

            boxes_tensor = torch.zeros(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (
                    0,
                ),
                dtype=torch.int64,
            )

            areas_tensor = torch.zeros(
                (
                    0,
                ),
                dtype=torch.float32,
            )

            iscrowd_tensor = torch.zeros(
                (
                    0,
                ),
                dtype=torch.int64,
            )

        else:
            masks_tensor = torch.as_tensor(
                np.stack(masks),
                dtype=torch.uint8,
            )

            boxes_tensor = torch.as_tensor(
                boxes,
                dtype=torch.float32,
            )

            labels_tensor = torch.as_tensor(
                labels,
                dtype=torch.int64,
            )

            areas_tensor = torch.as_tensor(
                areas,
                dtype=torch.float32,
            )

            iscrowd_tensor = torch.as_tensor(
                iscrowd,
                dtype=torch.int64,
            )

        image_tensor = TF.to_tensor(
            image
        )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return image_tensor, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    return tuple(
        zip(*batch)
    )


# ============================================================
# MEMBUAT MODEL
# ============================================================

def create_model():

    weights = (
        MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    model = maskrcnn_resnet50_fpn(
        weights=weights
    )

    # Ganti classifier box.
    in_features_box = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features_box,
            NUM_CLASSES,
        )
    )

    # Ganti classifier mask.
    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            NUM_CLASSES,
        )
    )

    return model


# ============================================================
# TRAINING SATU EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
):

    model.train()

    total_loss = 0.0

    for step, (images, targets) in enumerate(
        data_loader,
        start=1,
    ):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        loss_dict = model(
            images,
            targets,
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        loss_value = losses.item()

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += loss_value

        print(
            f"Epoch {epoch} | "
            f"Step {step}/{len(data_loader)} | "
            f"Loss: {loss_value:.6f}"
        )

    return total_loss / max(
        len(data_loader),
        1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 80)
    print("TRAINING MASK R-CNN")
    print("=" * 80)

    print(
        f"PyTorch       : {torch.__version__}"
    )

    print(
        f"Torchvision   : {torchvision.__version__}"
    )

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Dataset       : {DATASET_DIR}"
    )

    print()

    dataset = TreeInstanceDataset(
        IMAGE_DIR,
        MASK_DIR,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print(
        f"Jumlah citra  : {len(dataset)}"
    )

    print("[1/3] Membuat model Mask R-CNN...")

    model = create_model()

    model.to(
        DEVICE
    )

    print("Model berhasil dibuat.")

    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=8,
        gamma=0.1,
    )

    best_loss = float("inf")

    print("[2/3] Memulai training...")

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        start_time = time.time()

        average_loss = train_one_epoch(
            model,
            optimizer,
            data_loader,
            DEVICE,
            epoch,
        )

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"\nEpoch {epoch} selesai | "
            f"Average Loss: {average_loss:.6f} | "
            f"Waktu: {elapsed:.2f} detik"
        )

        if average_loss < best_loss:

            best_loss = average_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "num_classes": NUM_CLASSES,
                },
                MODEL_OUTPUT_PATH,
            )

            print(
                f"Best model disimpan: "
                f"{MODEL_OUTPUT_PATH}"
            )

        print("-" * 80)

    print("[3/3] Training selesai.")

    print("=" * 80)
    print("TRAINING MASK R-CNN BERHASIL")
    print("=" * 80)
    print(
        f"Best model: {MODEL_OUTPUT_PATH}"
    )
    print(
        f"Best loss : {best_loss:.6f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()