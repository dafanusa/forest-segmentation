import random
import csv
import json
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    TRAIN_PATTERN,
    MODEL_NAME,
    NUM_CLASSES,
    ID2LABEL,
    LABEL2ID,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    VAL_RATIO,
    NUM_WORKERS,
    SEED,
    DEVICE,
    CHECKPOINT_DIR,
    create_directories,
)

from dataset_loader import TCDParquetDataset


# ============================================================
# EARLY STOPPING CONFIGURATION
# ============================================================

EARLY_STOPPING_PATIENCE = 3

# Minimal perubahan validation loss agar dianggap membaik.
# Contoh: val_loss lama 0.5000, val_loss baru 0.49995
# tidak dianggap sebagai peningkatan berarti.
MIN_DELTA = 1e-4

# Gradient clipping untuk menjaga training tetap stabil.
MAX_GRAD_NORM = 1.0


# ============================================================
# SET SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# EARLY STOPPING CLASS
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=3,
        min_delta=1e-4,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):

        # Jika validation loss membaik secara signifikan
        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

            return True

        # Jika tidak ada peningkatan
        self.counter += 1

        if self.counter >= self.patience:

            self.early_stop = True

        return False


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    print(
        f"Riwayat training disimpan: {output_path}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SEED)
    create_directories()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("TRAINING SEGFORMER-B5 - TCD PARQUET")
    print("DENGAN EARLY STOPPING PATIENCE = 3")
    print("=" * 75)

    print(f"Device              : {DEVICE}")
    print(f"Dataset root        : {DATASET_ROOT}")
    print(f"Model                : {MODEL_NAME}")
    print(f"Image size           : {IMAGE_SIZE}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Jumlah epoch maksimal: {NUM_EPOCHS}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Weight decay         : {WEIGHT_DECAY}")
    print(f"Early stopping       : {EARLY_STOPPING_PATIENCE} epoch")
    print(f"Minimum delta        : {MIN_DELTA}")
    print("=" * 75)

    # ========================================================
    # LOAD PARQUET
    # ========================================================

    print("\n[1] Memuat dataset parquet...")

    train_files = sorted(
        str(path)
        for path in DATASET_ROOT.glob(
            TRAIN_PATTERN
        )
    )

    if not train_files:

        raise FileNotFoundError(
            "File train-*.parquet tidak ditemukan."
        )

    print(
        f"Jumlah file train: {len(train_files)}"
    )

    for file in train_files:
        print(f"  - {file}")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
        },
    )["train"]

    print(
        f"Jumlah data: {len(dataset)}"
    )

    print(
        f"Kolom dataset: {dataset.column_names}"
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

    print("\n[2] Memuat image processor...")

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        do_reduce_labels=False,
        size={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
        },
    )

    # ========================================================
    # DATASET WRAPPER
    # ========================================================

    print("\n[3] Menyiapkan dataset wrapper...")

    full_dataset = TCDParquetDataset(
        hf_dataset=dataset,
        processor=processor,
        image_size=IMAGE_SIZE,
    )

    val_size = int(
        len(full_dataset) * VAL_RATIO
    )

    train_size = len(full_dataset) - val_size

    if train_size <= 0 or val_size <= 0:

        raise ValueError(
            "Ukuran train atau validation tidak valid. "
            "Periksa VAL_RATIO."
        )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
        ],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Jumlah data training   : {len(train_dataset)}"
    )

    print(
        f"Jumlah data validation : {len(val_dataset)}"
    )

    print(
        f"Jumlah batch training  : {len(train_loader)}"
    )

    print(
        f"Jumlah batch validation: {len(val_loader)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n[4] Memuat model SegFormer-B5...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    use_amp = torch.cuda.is_available()

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    print(
        f"Mixed precision AMP: {use_amp}"
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
    )

    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    best_val_loss = float("inf")
    best_epoch = 0

    training_history = []

    best_path = CHECKPOINT_DIR / "best_model"
    last_path = CHECKPOINT_DIR / "last_model"

    history_path = (
        CHECKPOINT_DIR / "training_history.csv"
    )

    summary_path = (
        CHECKPOINT_DIR / "training_summary.json"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n[5] Memulai proses training...\n")

    for epoch in range(NUM_EPOCHS):

        epoch_number = epoch + 1

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch_number}/{NUM_EPOCHS} "
                "[TRAIN]"
            ),
        )

        for batch in train_bar:

            pixel_values = (
                batch["pixel_values"]
                .to(DEVICE)
            )

            labels = (
                batch["labels"]
                .to(DEVICE)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss = outputs.loss

            scaler.scale(loss).backward()

            # Unscale gradient sebelum clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= max(
            len(train_loader),
            1,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        val_bar = tqdm(
            val_loader,
            desc=(
                f"Epoch {epoch_number}/{NUM_EPOCHS} "
                "[VAL]"
            ),
        )

        with torch.no_grad():

            for batch in val_bar:

                pixel_values = (
                    batch["pixel_values"]
                    .to(DEVICE)
                )

                labels = (
                    batch["labels"]
                    .to(DEVICE)
                )

                with torch.cuda.amp.autocast(
                    enabled=use_amp
                ):

                    outputs = model(
                        pixel_values=pixel_values,
                        labels=labels,
                    )

                    loss = outputs.loss

                val_loss += loss.item()

                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        val_loss /= max(
            len(val_loader),
            1,
        )

        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # CEK BEST MODEL
        # ----------------------------------------------------

        is_best = early_stopping(
            val_loss
        )

        if is_best:

            best_val_loss = val_loss
            best_epoch = epoch_number

            model.save_pretrained(
                best_path
            )

            processor.save_pretrained(
                best_path
            )

            torch.save(
                {
                    "epoch": epoch_number,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "best_val_loss": best_val_loss,
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                },
                CHECKPOINT_DIR
                / "training_state.pth",
            )

            print(
                f"\n[✓] Best model disimpan "
                f"pada epoch {epoch_number}"
            )

        # ----------------------------------------------------
        # SAVE LAST MODEL
        # ----------------------------------------------------

        model.save_pretrained(
            last_path
        )

        processor.save_pretrained(
            last_path
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history_row = {
            "epoch": epoch_number,
            "train_loss": round(
                train_loss,
                6,
            ),
            "val_loss": round(
                val_loss,
                6,
            ),
            "best_val_loss": round(
                best_val_loss,
                6,
            ),
            "learning_rate": current_lr,
            "no_improvement_epochs": (
                early_stopping.counter
            ),
        }

        training_history.append(
            history_row
        )

        save_training_history(
            training_history,
            history_path,
        )

        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print("\n" + "-" * 75)

        print(
            f"Epoch {epoch_number}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss          : {train_loss:.6f}"
        )

        print(
            f"Val Loss            : {val_loss:.6f}"
        )

        print(
            f"Best Val Loss       : {best_val_loss:.6f}"
        )

        print(
            f"Best Epoch          : {best_epoch}"
        )

        print(
            f"Tidak membaik       : "
            f"{early_stopping.counter}/"
            f"{EARLY_STOPPING_PATIENCE} epoch"
        )

        print(
            f"Learning Rate       : {current_lr:.8f}"
        )

        print("-" * 75)

        # ----------------------------------------------------
        # EARLY STOPPING CHECK
        # ----------------------------------------------------

        if early_stopping.early_stop:

            print("\n" + "!" * 75)

            print(
                "EARLY STOPPING AKTIF"
            )

            print(
                f"Validation loss tidak membaik "
                f"selama {EARLY_STOPPING_PATIENCE} epoch berturut-turut."
            )

            print(
                f"Training dihentikan pada epoch "
                f"{epoch_number}."
            )

            print(
                f"Best model berasal dari epoch "
                f"{best_epoch}."
            )

            print("!" * 75)

            break

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    actual_epochs = len(
        training_history
    )

    training_summary = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": NUM_EPOCHS,
        "actual_epochs": actual_epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "min_delta": MIN_DELTA,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopped": (
            early_stopping.early_stop
        ),
        "device": str(DEVICE),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            training_summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 75)
    print("TRAINING SELESAI")
    print("=" * 75)

    print(
        f"Total epoch dijalankan: {actual_epochs}"
    )

    print(
        f"Best epoch            : {best_epoch}"
    )

    print(
        f"Best validation loss  : {best_val_loss:.6f}"
    )

    print(
        f"Best model            : {best_path}"
    )

    print(
        f"Last model            : {last_path}"
    )

    print(
        f"Training history      : {history_path}"
    )

    print(
        f"Training summary      : {summary_path}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()