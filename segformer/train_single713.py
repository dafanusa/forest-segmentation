from __future__ import annotations

from pathlib import Path
import random
import warnings

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from tqdm import tqdm


# ============================================================
# KONFIGURASI
# ============================================================

# Root project:
# D:\MAHASISWA\SEMESTER 5\FUNGSIONAL\PRAKTIKUM\STRD-Net

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    BASE_DIR
    / "tcd_dataset"
    / "dataset"
    / "713.tif"
)

MASK_PATH = (
    BASE_DIR
    / "tcd_dataset"
    / "dataset"
    / "semantic_mask.tif"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs_single_713"
)

BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Model ringan untuk pengujian single image
MODEL_NAME = "nvidia/mit-b0"

# Jumlah kelas:
# 0 = background
# 1 = tree
NUM_CLASSES = 2

BACKGROUND_CLASS = 0
TREE_CLASS = 1

PATCH_SIZE = 512

NUM_EPOCHS = 10

BATCH_SIZE = 1

LEARNING_RATE = 5e-5

VAL_RATIO = 0.2

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# UTILITAS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# DATASET PATCH
# ============================================================

class SingleImagePatchDataset(Dataset):
    """
    Dataset untuk memotong satu citra besar menjadi patch.

    Citra:
        [C, H, W]

    Mask:
        [H, W]

    Label:
        0 = background
        1 = tree
    """

    def __init__(
        self,
        image_path: Path,
        mask_path: Path,
        processor: SegformerImageProcessor,
        patch_size: int = 512,
    ):
        self.image_path = image_path
        self.mask_path = mask_path
        self.processor = processor
        self.patch_size = patch_size

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Citra tidak ditemukan:\n{self.image_path}"
            )

        if not self.mask_path.exists():
            raise FileNotFoundError(
                f"Mask tidak ditemukan:\n{self.mask_path}"
            )

        print_header("MEMBACA DATASET")

        print(f"Image path : {self.image_path.resolve()}")
        print(f"Mask path  : {self.mask_path.resolve()}")

        # ----------------------------------------------------
        # Membaca citra
        # ----------------------------------------------------
        with rasterio.open(self.image_path) as src:
            image = src.read()
            self.image_profile = src.profile

        # ----------------------------------------------------
        # Membaca mask
        # ----------------------------------------------------
        with rasterio.open(self.mask_path) as src:
            mask = src.read(1)

        print(f"Ukuran citra awal : {image.shape}")
        print(f"Ukuran mask awal  : {mask.shape}")

        if image.ndim != 3:
            raise ValueError(
                "Citra harus memiliki format [C, H, W]. "
                f"Didapatkan: {image.shape}"
            )

        if image.shape[1:] != mask.shape:
            raise ValueError(
                "Ukuran citra dan mask tidak sama.\n"
                f"Ukuran citra: {image.shape[1:]}\n"
                f"Ukuran mask : {mask.shape}"
            )

        # ----------------------------------------------------
        # Memastikan citra memiliki 3 channel
        # ----------------------------------------------------
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)

        elif image.shape[0] >= 3:
            image = image[:3]

        else:
            raise ValueError(
                f"Jumlah channel citra tidak didukung: "
                f"{image.shape[0]}"
            )

        # ----------------------------------------------------
        # Normalisasi citra menjadi uint8 0-255
        # ----------------------------------------------------
        image = image.astype(np.float32)

        image_min = np.nanmin(image)
        image_max = np.nanmax(image)

        print(f"Nilai minimum citra : {image_min}")
        print(f"Nilai maksimum citra: {image_max}")

        if image_max > image_min:
            image = (
                (image - image_min)
                / (image_max - image_min)
                * 255.0
            )
        else:
            image = np.zeros_like(image)

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Normalisasi mask
        # ----------------------------------------------------
        mask = mask.astype(np.int64)

        unique_values_before = np.unique(mask)

        print(
            "Nilai unik mask sebelum normalisasi: "
            f"{unique_values_before}"
        )

        # Semua nilai > 0 dianggap sebagai tree
        mask = np.where(
            mask > 0,
            TREE_CLASS,
            BACKGROUND_CLASS,
        ).astype(np.int64)

        unique_values_after = np.unique(mask)

        print(
            "Nilai unik mask setelah normalisasi : "
            f"{unique_values_after}"
        )

        # ----------------------------------------------------
        # Statistik mask
        # ----------------------------------------------------
        tree_pixels = np.sum(
            mask == TREE_CLASS
        )

        background_pixels = np.sum(
            mask == BACKGROUND_CLASS
        )

        total_pixels = mask.size

        tree_percentage = (
            tree_pixels / total_pixels
        ) * 100.0

        background_percentage = (
            background_pixels / total_pixels
        ) * 100.0

        print(
            f"Background : "
            f"{background_percentage:.2f}%"
        )

        print(
            f"Tree       : "
            f"{tree_percentage:.2f}%"
        )

        if tree_pixels == 0:
            warnings.warn(
                "Mask tidak memiliki piksel tree. "
                "Model tidak dapat belajar kelas tree."
            )

        self.image = image
        self.mask = mask

        # ----------------------------------------------------
        # Membuat koordinat patch
        # ----------------------------------------------------
        height, width = mask.shape

        self.patch_coordinates = []

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                x_end = min(
                    x + patch_size,
                    width,
                )

                y_end = min(
                    y + patch_size,
                    height,
                )

                patch_width = x_end - x
                patch_height = y_end - y

                # Lewati patch yang terlalu kecil
                if patch_width < 32 or patch_height < 32:
                    continue

                self.patch_coordinates.append(
                    (
                        x,
                        y,
                        x_end,
                        y_end,
                    )
                )

        print(
            f"Jumlah patch yang dibuat: "
            f"{len(self.patch_coordinates)}"
        )

        if len(self.patch_coordinates) < 2:
            raise RuntimeError(
                "Jumlah patch kurang dari 2. "
                "Tidak dapat melakukan pembagian training "
                "dan validation."
            )

    def __len__(self):
        return len(self.patch_coordinates)

    def __getitem__(self, index: int):
        x_start, y_start, x_end, y_end = (
            self.patch_coordinates[index]
        )

        image_patch = self.image[
            :,
            y_start:y_end,
            x_start:x_end,
        ]

        mask_patch = self.mask[
            y_start:y_end,
            x_start:x_end,
        ]

        # [C, H, W] menjadi [H, W, C]
        image_patch = np.transpose(
            image_patch,
            (1, 2, 0),
        )

        encoded = self.processor(
            images=image_patch,
            segmentation_maps=mask_patch,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded[
                "pixel_values"
            ].squeeze(0),

            "labels": encoded[
                "labels"
            ].squeeze(0),
        }


# ============================================================
# PERHITUNGAN METRIK
# ============================================================

def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Menghitung:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Perhitungan dilakukan pada level piksel.
    """

    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if predictions.shape != labels.shape:
        raise ValueError(
            "Ukuran predictions dan labels berbeda.\n"
            f"Predictions: {predictions.shape}\n"
            f"Labels     : {labels.shape}"
        )

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # Abaikan label 255 jika ada
    valid = labels != 255

    predictions = predictions[valid]
    labels = labels[valid]

    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    true_positive = np.sum(
        (predictions == TREE_CLASS)
        & (labels == TREE_CLASS)
    )

    true_negative = np.sum(
        (predictions == BACKGROUND_CLASS)
        & (labels == BACKGROUND_CLASS)
    )

    false_positive = np.sum(
        (predictions == TREE_CLASS)
        & (labels == BACKGROUND_CLASS)
    )

    false_negative = np.sum(
        (predictions == BACKGROUND_CLASS)
        & (labels == TREE_CLASS)
    )

    accuracy = (
        true_positive + true_negative
    ) / max(len(labels), 1)

    precision = true_positive / max(
        true_positive + false_positive,
        1,
    )

    recall = true_positive / max(
        true_positive + false_negative,
        1,
    )

    f1 = (
        2.0 * precision * recall
    ) / max(
        precision + recall,
        1e-8,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# TRAINING SATU EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch_number,
):
    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch_number}",
        leave=True,
    )

    for batch in progress_bar:
        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Perbaikan penting:
        # Output SegFormer lebih kecil daripada label.
        # Resize logits ke ukuran label sebelum argmax.
        # ----------------------------------------------------
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_predictions.append(
            predictions.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        running_loss
        / max(len(dataloader), 1)
    )

    all_predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    all_labels = torch.cat(
        all_labels,
        dim=0,
    )

    metrics = calculate_metrics(
        all_predictions,
        all_labels,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# VALIDASI SATU EPOCH
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    device,
    epoch_number,
):
    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Validation Epoch {epoch_number}",
        leave=True,
    )

    for batch in progress_bar:
        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        running_loss += loss.item()

        # Resize logits ke ukuran label
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_predictions.append(
            predictions.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        running_loss
        / max(len(dataloader), 1)
    )

    all_predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    all_labels = torch.cat(
        all_labels,
        dim=0,
    )

    metrics = calculate_metrics(
        all_predictions,
        all_labels,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    set_seed(SEED)

    print_header(
        "KONFIGURASI TRAINING SEGFORMER SINGLE IMAGE"
    )

    print(f"Base directory : {BASE_DIR}")
    print(
        f"Image path     : "
        f"{IMAGE_PATH.resolve()}"
    )
    print(
        f"Mask path      : "
        f"{MASK_PATH.resolve()}"
    )
    print(
        f"Output dir     : "
        f"{OUTPUT_DIR.resolve()}"
    )
    print(f"Model          : {MODEL_NAME}")
    print(f"Device         : {DEVICE}")
    print(f"Patch size     : {PATCH_SIZE}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Epoch          : {NUM_EPOCHS}")
    print(f"Learning rate  : {LEARNING_RATE}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------
    print_header("MEMUAT PROCESSOR")

    processor = (
        SegformerImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = SingleImagePatchDataset(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        processor=processor,
        patch_size=PATCH_SIZE,
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------
    total_size = len(dataset)

    val_size = max(
        1,
        int(total_size * VAL_RATIO),
    )

    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size,
        ],
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    print_header("PEMBAGIAN DATASET")

    print(f"Total patch      : {total_size}")
    print(
        f"Training patch   : "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation patch : "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    print_header("MEMUAT MODEL SEGFORMER")

    model = (
        SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
        )
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_val_f1 = -1.0
    best_val_loss = float("inf")

    history = []

    print_header("MULAI TRAINING")

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n" + "-" * 80)
        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )
        print("-" * 80)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch_number=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=DEVICE,
            epoch_number=epoch,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,

            "train_loss": train_metrics[
                "loss"
            ],

            "train_accuracy": train_metrics[
                "accuracy"
            ],

            "train_precision": train_metrics[
                "precision"
            ],

            "train_recall": train_metrics[
                "recall"
            ],

            "train_f1": train_metrics[
                "f1"
            ],

            "val_loss": val_metrics[
                "loss"
            ],

            "val_accuracy": val_metrics[
                "accuracy"
            ],

            "val_precision": val_metrics[
                "precision"
            ],

            "val_recall": val_metrics[
                "recall"
            ],

            "val_f1": val_metrics[
                "f1"
            ],
        }

        history.append(history_row)

        print("\nHASIL TRAINING")
        print(
            f"Learning rate : "
            f"{current_lr:.8f}"
        )

        print("\nTraining:")
        print(
            f"  Loss      : "
            f"{train_metrics['loss']:.6f}"
        )
        print(
            f"  Accuracy  : "
            f"{train_metrics['accuracy']:.6f}"
        )
        print(
            f"  Precision : "
            f"{train_metrics['precision']:.6f}"
        )
        print(
            f"  Recall    : "
            f"{train_metrics['recall']:.6f}"
        )
        print(
            f"  F1-score  : "
            f"{train_metrics['f1']:.6f}"
        )

        print("\nValidation:")
        print(
            f"  Loss      : "
            f"{val_metrics['loss']:.6f}"
        )
        print(
            f"  Accuracy  : "
            f"{val_metrics['accuracy']:.6f}"
        )
        print(
            f"  Precision : "
            f"{val_metrics['precision']:.6f}"
        )
        print(
            f"  Recall    : "
            f"{val_metrics['recall']:.6f}"
        )
        print(
            f"  F1-score  : "
            f"{val_metrics['f1']:.6f}"
        )

        # ----------------------------------------------------
        # Simpan checkpoint setiap epoch
        # ----------------------------------------------------
        epoch_checkpoint_dir = (
            CHECKPOINT_DIR
            / f"epoch_{epoch:03d}"
        )

        epoch_checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            epoch_checkpoint_dir
        )

        processor.save_pretrained(
            epoch_checkpoint_dir
        )

        # ----------------------------------------------------
        # Simpan model terbaik
        # ----------------------------------------------------
        is_better = (
            val_metrics["f1"] > best_val_f1
            or (
                val_metrics["f1"] == best_val_f1
                and val_metrics["loss"]
                < best_val_loss
            )
        )

        if is_better:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]

            model.save_pretrained(
                BEST_MODEL_DIR
            )

            processor.save_pretrained(
                BEST_MODEL_DIR
            )

            print(
                "\nModel terbaik berhasil disimpan di:"
            )

            print(
                BEST_MODEL_DIR.resolve()
            )

    # --------------------------------------------------------
    # Simpan model final
    # --------------------------------------------------------
    print_header("MENYIMPAN MODEL FINAL")

    model.save_pretrained(
        FINAL_MODEL_DIR
    )

    processor.save_pretrained(
        FINAL_MODEL_DIR
    )

    print(
        "Model final disimpan di:"
    )

    print(
        FINAL_MODEL_DIR.resolve()
    )

    # --------------------------------------------------------
    # Simpan riwayat training
    # --------------------------------------------------------
    history_path = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    try:
        import pandas as pd

        history_df = pd.DataFrame(history)

        history_df.to_csv(
            history_path,
            index=False,
        )

        print(
            "\nRiwayat training disimpan di:"
        )

        print(
            history_path.resolve()
        )

    except ImportError:
        print(
            "\nPandas tidak tersedia. "
            "Riwayat training tidak disimpan."
        )

    # --------------------------------------------------------
    # Ringkasan akhir
    # --------------------------------------------------------
    print_header("TRAINING SELESAI")

    print(
        f"Best validation F1 : "
        f"{best_val_f1:.6f}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print("\nFolder output:")
    print(OUTPUT_DIR.resolve())

    print("\nModel terbaik:")
    print(BEST_MODEL_DIR.resolve())

    print("\nModel final:")
    print(FINAL_MODEL_DIR.resolve())

    print("\nFile model yang diharapkan:")
    print("  config.json")
    print("  preprocessor_config.json")
    print(
        "  model.safetensors atau "
        "pytorch_model.bin"
    )


if __name__ == "__main__":
    main()