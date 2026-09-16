from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from config import (
    DATASET_ROOT,
    TRAIN_PATTERN,
    MODEL_NAME,
    NUM_CLASSES,
    ID2LABEL,
    LABEL2ID,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_RATIO,
    NUM_WORKERS,
    SEED,
    DEVICE,
    CHECKPOINT_DIR,
)

from dataset_loader import TCDParquetDataset


# ============================================================
# KONFIGURASI OUTPUT
# ============================================================

MODEL_PATH = CHECKPOINT_DIR / "best_model"

EVALUATION_OUTPUT_DIR = Path(
    "outputs/evaluation"
)

EVALUATION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Label ignore yang umum digunakan oleh SegFormer
IGNORE_INDEX = 255


# ============================================================
# SET SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MEMBUAT CONFUSION MATRIX MANUAL
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    predictions,
    labels,
    num_classes,
    ignore_index=255,
):

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = (
        labels != ignore_index
    )

    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    valid_class_mask = (
        (labels >= 0)
        & (labels < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )

    predictions = predictions[valid_class_mask]
    labels = labels[valid_class_mask]

    for true_label, predicted_label in zip(
        labels,
        predictions,
    ):

        confusion_matrix[
            int(true_label),
            int(predicted_label),
        ] += 1

    return confusion_matrix


# ============================================================
# MENGHITUNG METRIK DARI CONFUSION MATRIX
# ============================================================

def calculate_metrics_from_confusion_matrix(
    confusion_matrix,
    id2label,
):

    num_classes = confusion_matrix.shape[0]

    total_pixels = np.sum(
        confusion_matrix
    )

    correct_pixels = np.trace(
        confusion_matrix
    )

    pixel_accuracy = (
        correct_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    per_class_metrics = []

    iou_values = []
    dice_values = []
    precision_values = []
    recall_values = []
    f1_values = []

    for class_id in range(num_classes):

        true_positive = confusion_matrix[
            class_id,
            class_id,
        ]

        false_positive = (
            np.sum(
                confusion_matrix[
                    :,
                    class_id,
                ]
            )
            - true_positive
        )

        false_negative = (
            np.sum(
                confusion_matrix[
                    class_id,
                    :,
                ]
            )
            - true_positive
        )

        true_negative = (
            total_pixels
            - true_positive
            - false_positive
            - false_negative
        )

        precision = (
            true_positive
            / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive
            / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        iou = (
            true_positive
            / (
                true_positive
                + false_positive
                + false_negative
            )
            if (
                true_positive
                + false_positive
                + false_negative
            ) > 0
            else 0.0
        )

        dice = (
            2 * true_positive
            / (
                2 * true_positive
                + false_positive
                + false_negative
            )
            if (
                2 * true_positive
                + false_positive
                + false_negative
            ) > 0
            else 0.0
        )

        specificity = (
            true_negative
            / (
                true_negative
                + false_positive
            )
            if (
                true_negative
                + false_positive
            ) > 0
            else 0.0
        )

        support = int(
            np.sum(
                confusion_matrix[
                    class_id,
                    :,
                ]
            )
        )

        class_name = id2label.get(
            class_id,
            f"class_{class_id}",
        )

        per_class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "iou": float(iou),
                "dice": float(dice),
                "specificity": float(specificity),
                "support": support,
            }
        )

        iou_values.append(iou)
        dice_values.append(dice)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)

    mean_iou = float(
        np.mean(iou_values)
    )

    mean_dice = float(
        np.mean(dice_values)
    )

    macro_precision = float(
        np.mean(precision_values)
    )

    macro_recall = float(
        np.mean(recall_values)
    )

    macro_f1 = float(
        np.mean(f1_values)
    )

    frequency_weighted_iou = 0.0

    if total_pixels > 0:

        for class_id in range(num_classes):

            class_support = np.sum(
                confusion_matrix[
                    class_id,
                    :,
                ]
            )

            class_iou = iou_values[
                class_id
            ]

            frequency = (
                class_support
                / total_pixels
            )

            frequency_weighted_iou += (
                frequency * class_iou
            )

    metrics = {
        "pixel_accuracy": float(
            pixel_accuracy
        ),
        "mean_iou": mean_iou,
        "frequency_weighted_iou": float(
            frequency_weighted_iou
        ),
        "mean_dice": mean_dice,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "total_valid_pixels": int(
            total_pixels
        ),
        "correct_pixels": int(
            correct_pixels
        ),
        "per_class": per_class_metrics,
        "confusion_matrix": (
            confusion_matrix.tolist()
        ),
    }

    return metrics


# ============================================================
# EVALUASI SATU DATASET
# ============================================================

def evaluate_dataset(
    model,
    dataloader,
    dataset_name,
):

    print()
    print("=" * 75)
    print(
        f"EVALUASI DATASET: {dataset_name.upper()}"
    )
    print("=" * 75)

    model.eval()

    total_loss = 0.0
    total_batches = 0

    confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    total_samples = 0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluasi {dataset_name}",
        )

        for batch in progress_bar:

            pixel_values = batch[
                "pixel_values"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=(
                    labels.shape[-2],
                    labels.shape[-1],
                ),
                mode="bilinear",
                align_corners=False,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            confusion_matrix = (
                update_confusion_matrix(
                    confusion_matrix,
                    predictions.cpu().numpy(),
                    labels.cpu().numpy(),
                    NUM_CLASSES,
                    IGNORE_INDEX,
                )
            )

            total_samples += (
                pixel_values.shape[0]
            )

    average_loss = (
        total_loss / total_batches
        if total_batches > 0
        else 0.0
    )

    metrics = (
        calculate_metrics_from_confusion_matrix(
            confusion_matrix,
            ID2LABEL,
        )
    )

    metrics["dataset"] = dataset_name
    metrics["average_loss"] = float(
        average_loss
    )
    metrics["total_samples"] = int(
        total_samples
    )

    return metrics


# ============================================================
# MENYIMPAN METRIK JSON
# ============================================================

def save_metrics_json(
    metrics,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"JSON evaluasi disimpan: {output_path}"
    )


# ============================================================
# MENYIMPAN METRIK CSV
# ============================================================

def save_metrics_csv(
    metrics,
    output_path,
):

    rows = metrics["per_class"]

    if len(rows) == 0:
        return

    fieldnames = [
        "class_id",
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "dice",
        "specificity",
        "support",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"CSV metrik per kelas disimpan: {output_path}"
    )


# ============================================================
# MENYIMPAN RINGKASAN METRIK
# ============================================================

def save_summary_csv(
    all_metrics,
    output_path,
):

    rows = []

    for metrics in all_metrics:

        rows.append(
            {
                "dataset": metrics["dataset"],
                "total_samples": metrics[
                    "total_samples"
                ],
                "average_loss": metrics[
                    "average_loss"
                ],
                "pixel_accuracy": metrics[
                    "pixel_accuracy"
                ],
                "mean_iou": metrics[
                    "mean_iou"
                ],
                "frequency_weighted_iou": metrics[
                    "frequency_weighted_iou"
                ],
                "mean_dice": metrics[
                    "mean_dice"
                ],
                "macro_precision": metrics[
                    "macro_precision"
                ],
                "macro_recall": metrics[
                    "macro_recall"
                ],
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "total_valid_pixels": metrics[
                    "total_valid_pixels"
                ],
                "correct_pixels": metrics[
                    "correct_pixels"
                ],
            }
        )

    if len(rows) == 0:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"CSV ringkasan disimpan: {output_path}"
    )


# ============================================================
# MENYIMPAN CONFUSION MATRIX CSV
# ============================================================

def save_confusion_matrix_csv(
    metrics,
    output_path,
):

    confusion_matrix = np.array(
        metrics["confusion_matrix"]
    )

    class_names = []

    for class_id in range(NUM_CLASSES):

        class_names.append(
            ID2LABEL.get(
                class_id,
                f"class_{class_id}",
            )
        )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "True/Predicted"
            ]
            + class_names
        )

        for index, row in enumerate(
            confusion_matrix
        ):

            writer.writerow(
                [
                    class_names[index]
                ]
                + row.tolist()
            )

    print(
        f"Confusion matrix disimpan: {output_path}"
    )


# ============================================================
# MEMBUAT GAMBAR CONFUSION MATRIX
# ============================================================

def save_confusion_matrix_image(
    metrics,
    output_path,
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Matplotlib belum terpasang. "
            "Confusion matrix PNG tidak dibuat."
        )

        return

    confusion_matrix = np.array(
        metrics["confusion_matrix"]
    )

    class_names = []

    for class_id in range(NUM_CLASSES):

        class_names.append(
            ID2LABEL.get(
                class_id,
                f"class_{class_id}",
            )
        )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    image = axis.imshow(
        confusion_matrix
    )

    figure.colorbar(image)

    axis.set_title(
        f"Confusion Matrix - "
        f"{metrics['dataset']}"
    )

    axis.set_xlabel(
        "Predicted Label"
    )

    axis.set_ylabel(
        "True Label"
    )

    axis.set_xticks(
        range(NUM_CLASSES)
    )

    axis.set_yticks(
        range(NUM_CLASSES)
    )

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        class_names
    )

    threshold = (
        confusion_matrix.max() / 2
        if confusion_matrix.size > 0
        else 0
    )

    for row in range(NUM_CLASSES):

        for col in range(NUM_CLASSES):

            value = confusion_matrix[
                row,
                col,
            ]

            axis.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Gambar confusion matrix disimpan: "
        f"{output_path}"
    )


# ============================================================
# MENAMPILKAN HASIL EVALUASI
# ============================================================

def print_metrics(metrics):

    print()
    print("-" * 75)
    print(
        f"HASIL EVALUASI: "
        f"{metrics['dataset'].upper()}"
    )
    print("-" * 75)

    print(
        f"Jumlah sampel          : "
        f"{metrics['total_samples']}"
    )

    print(
        f"Average loss           : "
        f"{metrics['average_loss']:.6f}"
    )

    print(
        f"Pixel accuracy         : "
        f"{metrics['pixel_accuracy'] * 100:.4f}%"
    )

    print(
        f"Mean IoU               : "
        f"{metrics['mean_iou'] * 100:.4f}%"
    )

    print(
        f"Frequency Weighted IoU : "
        f"{metrics['frequency_weighted_iou'] * 100:.4f}%"
    )

    print(
        f"Mean Dice              : "
        f"{metrics['mean_dice'] * 100:.4f}%"
    )

    print(
        f"Macro Precision        : "
        f"{metrics['macro_precision'] * 100:.4f}%"
    )

    print(
        f"Macro Recall           : "
        f"{metrics['macro_recall'] * 100:.4f}%"
    )

    print(
        f"Macro F1-score         : "
        f"{metrics['macro_f1'] * 100:.4f}%"
    )

    print()
    print("METRIK PER KELAS")
    print("-" * 75)

    print(
        f"{'Kelas':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'IoU':>12}"
        f"{'Dice':>12}"
    )

    for item in metrics["per_class"]:

        print(
            f"{item['class_name']:<20}"
            f"{item['precision'] * 100:>11.2f}%"
            f"{item['recall'] * 100:>11.2f}%"
            f"{item['f1_score'] * 100:>11.2f}%"
            f"{item['iou'] * 100:>11.2f}%"
            f"{item['dice'] * 100:>11.2f}%"
        )

    print("-" * 75)


# ============================================================
# LOAD DATASET DAN MODEL
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 75)
    print("EVALUASI MODEL SEGFORMER-B5")
    print("=" * 75)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Dataset root : {DATASET_ROOT}"
    )

    print(
        f"Model path   : {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan training sudah selesai "
            "dan best_model sudah tersimpan."
        )

    # --------------------------------------------------------
    # LOAD FILE PARQUET
    # --------------------------------------------------------

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

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
        },
    )["train"]

    print(
        f"Jumlah seluruh data: {len(dataset)}"
    )

    print(
        f"Kolom dataset: {dataset.column_names}"
    )

    # --------------------------------------------------------
    # LOAD PROCESSOR DARI MODEL TERLATIH
    # --------------------------------------------------------

    processor = (
        SegformerImageProcessor.from_pretrained(
            MODEL_PATH,
            do_reduce_labels=False,
            size={
                "height": IMAGE_SIZE,
                "width": IMAGE_SIZE,
            },
        )
    )

    # --------------------------------------------------------
    # DATASET WRAPPER
    # --------------------------------------------------------

    full_dataset = TCDParquetDataset(
        hf_dataset=dataset,
        processor=processor,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # PEMBAGIAN DATA HARUS SAMA DENGAN TRAINING
    # --------------------------------------------------------

    val_size = int(
        len(full_dataset) * VAL_RATIO
    )

    train_size = (
        len(full_dataset) - val_size
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
        ],
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    print(
        f"Train dataset: {len(train_dataset)}"
    )

    print(
        f"Val dataset  : {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # LOAD MODEL TERBAIK
    # --------------------------------------------------------

    print()
    print("Memuat model terbaik...")

    model = (
        SegformerForSemanticSegmentation.from_pretrained(
            MODEL_PATH
        )
    )

    model.to(DEVICE)
    model.eval()

    print(
        "Model berhasil dimuat."
    )

    # --------------------------------------------------------
    # EVALUASI TRAIN, VALIDATION, DAN SELURUH DATA
    # --------------------------------------------------------

    train_metrics = evaluate_dataset(
        model,
        train_loader,
        "train",
    )

    val_metrics = evaluate_dataset(
        model,
        val_loader,
        "validation",
    )

    all_metrics = evaluate_dataset(
        model,
        full_loader,
        "all_dataset",
    )

    # --------------------------------------------------------
    # TAMPILKAN HASIL
    # --------------------------------------------------------

    print_metrics(
        train_metrics
    )

    print_metrics(
        val_metrics
    )

    print_metrics(
        all_metrics
    )

    # --------------------------------------------------------
    # SIMPAN HASIL TRAIN
    # --------------------------------------------------------

    save_metrics_json(
        train_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_train.json",
    )

    save_metrics_csv(
        train_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_train_per_class.csv",
    )

    save_confusion_matrix_csv(
        train_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_train.csv",
    )

    save_confusion_matrix_image(
        train_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_train.png",
    )

    # --------------------------------------------------------
    # SIMPAN HASIL VALIDATION
    # --------------------------------------------------------

    save_metrics_json(
        val_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_validation.json",
    )

    save_metrics_csv(
        val_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_validation_per_class.csv",
    )

    save_confusion_matrix_csv(
        val_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_validation.csv",
    )

    save_confusion_matrix_image(
        val_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_validation.png",
    )

    # --------------------------------------------------------
    # SIMPAN HASIL SELURUH DATA
    # --------------------------------------------------------

    save_metrics_json(
        all_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_all_dataset.json",
    )

    save_metrics_csv(
        all_metrics,
        EVALUATION_OUTPUT_DIR
        / "metrics_all_dataset_per_class.csv",
    )

    save_confusion_matrix_csv(
        all_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_all_dataset.csv",
    )

    save_confusion_matrix_image(
        all_metrics,
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix_all_dataset.png",
    )

    # --------------------------------------------------------
    # SIMPAN RINGKASAN SEMUA DATASET
    # --------------------------------------------------------

    save_summary_csv(
        [
            train_metrics,
            val_metrics,
            all_metrics,
        ],
        EVALUATION_OUTPUT_DIR
        / "evaluation_summary.csv",
    )

    # --------------------------------------------------------
    # SELESAI
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("EVALUASI MODEL SELESAI")
    print("=" * 75)

    print(
        f"Folder output: "
        f"{EVALUATION_OUTPUT_DIR.resolve()}"
    )

    print()
    print("File hasil evaluasi:")

    print(
        "  - evaluation_summary.csv"
    )

    print(
        "  - metrics_train.json"
    )

    print(
        "  - metrics_validation.json"
    )

    print(
        "  - metrics_all_dataset.json"
    )

    print(
        "  - metrics_train_per_class.csv"
    )

    print(
        "  - metrics_validation_per_class.csv"
    )

    print(
        "  - metrics_all_dataset_per_class.csv"
    )

    print(
        "  - confusion_matrix_train.png"
    )

    print(
        "  - confusion_matrix_validation.png"
    )

    print(
        "  - confusion_matrix_all_dataset.png"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()