import random
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


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(SEED)
    create_directories()

    print("=" * 70)
    print("TRAINING SEGFORMER-B5 - TCD PARQUET")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")

    # --------------------------------------------------------
    # LOAD PARQUET
    # --------------------------------------------------------

    train_files = sorted(
        str(p)
        for p in DATASET_ROOT.glob(
            TRAIN_PATTERN
        )
    )

    if not train_files:
        raise FileNotFoundError(
            "File train-*.parquet tidak ditemukan."
        )

    print(f"Jumlah file train: {len(train_files)}")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
        },
    )["train"]

    print(f"Jumlah data: {len(dataset)}")
    print(f"Kolom: {dataset.column_names}")

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        do_reduce_labels=False,
        size={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
        },
    )

    # --------------------------------------------------------
    # DATASET WRAPPER
    # --------------------------------------------------------

    full_dataset = TCDParquetDataset(
        hf_dataset=dataset,
        processor=processor,
        image_size=IMAGE_SIZE,
    )

    val_size = int(
        len(full_dataset) * VAL_RATIO
    )

    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
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

    print(f"Train: {len(train_dataset)}")
    print(f"Val  : {len(val_dataset)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]",
        )

        for batch in train_bar:

            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= max(len(train_loader), 1)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [VAL]",
            ):

                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                val_loss += outputs.loss.item()

        val_loss /= max(len(val_loader), 1)

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = CHECKPOINT_DIR / "best_model"

            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                CHECKPOINT_DIR / "training_state.pth",
            )

            print(
                f"Best model disimpan: {best_path}"
            )

    print("\nTraining selesai.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()