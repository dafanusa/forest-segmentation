from __future__ import annotations

from pathlib import Path
import sys
import json
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# IMPORT CONFIG
# ============================================================

try:
    from .config import (
        NUM_CLASSES,
        EVAL_GT_DIR,
        EVAL_PRED_DIR,
        EVAL_OUTPUT_DIR,
        EVAL_METRICS_JSON,
        EVAL_METRICS_CSV,
        EVAL_CONFUSION_MATRIX_CSV,
        EVAL_CONFUSION_MATRIX_PNG,
        EVAL_CLASS_NAMES,
    )

except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from upernet.config import (
        NUM_CLASSES,
        EVAL_GT_DIR,
        EVAL_PRED_DIR,
        EVAL_OUTPUT_DIR,
        EVAL_METRICS_JSON,
        EVAL_METRICS_CSV,
        EVAL_CONFUSION_MATRIX_CSV,
        EVAL_CONFUSION_MATRIX_PNG,
        EVAL_CLASS_NAMES,
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

EVAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# KONFIGURASI
# ============================================================

PREDICTION_SUFFIX = "_pred"
GROUND_TRUTH_SUFFIX = "_mask"

# Nilai yang diabaikan ketika evaluasi
IGNORE_INDEX = 255


# ============================================================
# AMBIL FILE PREDIKSI
# ============================================================

def get_prediction_files(
    prediction_dir: Path,
) -> list[Path]:
    """
    Mengambil seluruh file hasil prediksi UPerNet.
    """

    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Folder prediksi tidak ditemukan:\n"
            f"{prediction_dir}"
        )

    prediction_files = sorted(
        list(prediction_dir.glob("*.tif"))
        + list(prediction_dir.glob("*.tiff"))
    )

    prediction_files = [
        file
        for file in prediction_files
        if file.stem.endswith(PREDICTION_SUFFIX)
    ]

    if len(prediction_files) == 0:
        raise FileNotFoundError(
            f"Tidak ditemukan file prediksi *_pred.tif di:\n"
            f"{prediction_dir}"
        )

    return prediction_files


# ============================================================
# CARI GROUND-TRUTH
# ============================================================

def find_ground_truth(
    prediction_path: Path,
) -> Path | None:
    """
    Contoh:
        100_pred.tif

    Akan mencari:
        100_mask.tif
        100.tif
    """

    prediction_stem = prediction_path.stem

    if prediction_stem.endswith(PREDICTION_SUFFIX):
        image_stem = prediction_stem[
            : -len(PREDICTION_SUFFIX)
        ]
    else:
        image_stem = prediction_stem

    candidate_paths = [
        EVAL_GT_DIR / f"{image_stem}{GROUND_TRUTH_SUFFIX}.tif",
        EVAL_GT_DIR / f"{image_stem}{GROUND_TRUTH_SUFFIX}.tiff",
        EVAL_GT_DIR / f"{image_stem}.tif",
        EVAL_GT_DIR / f"{image_stem}.tiff",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return None


# ============================================================
# BACA MASK
# ============================================================

def read_mask(
    mask_path: Path,
) -> tuple[np.ndarray, dict]:
    """
    Membaca band pertama dari raster mask.
    """

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": str(src.transform),
            "nodata": src.nodata,
        }

    return mask, metadata


# ============================================================
# NORMALISASI MASK
# ============================================================

def normalize_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Mengubah mask menjadi kelas:
        0 = background
        1 = tree

    Mendukung:
        0 dan 1
        0 dan 255
    """

    mask = mask.astype(np.int64)

    unique_values = np.unique(mask)

    # Untuk mask 0 dan 255:
    # 255 dianggap sebagai kelas tree
    if 255 in unique_values:
        mask = np.where(
            mask == 255,
            1,
            mask,
        )

    # Karena model adalah segmentasi biner,
    # seluruh nilai positif dianggap tree
    if NUM_CLASSES == 2:
        mask = np.where(
            mask > 0,
            1,
            0,
        )

    return mask


# ============================================================
# UPDATE CONFUSION MATRIX
# ============================================================

def update_confusion_matrix(
    global_cm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Menambahkan confusion matrix satu file
    ke confusion matrix keseluruhan.
    """

    valid = (
        (y_true >= 0)
        & (y_true < NUM_CLASSES)
        & (y_pred >= 0)
        & (y_pred < NUM_CLASSES)
    )

    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]

    cm = confusion_matrix(
        y_true_valid,
        y_pred_valid,
        labels=list(range(NUM_CLASSES)),
    )

    global_cm += cm


# ============================================================
# HITUNG METRIK
# ============================================================

def calculate_metrics(
    cm: np.ndarray,
) -> dict:
    """
    Menghitung metrik berdasarkan confusion matrix global.
    """

    total_pixels = int(cm.sum())

    accuracy = (
        np.trace(cm) / total_pixels
        if total_pixels > 0
        else 0.0
    )

    per_class = {}

    precision_values = []
    recall_values = []
    f1_values = []
    iou_values = []
    dice_values = []

    for class_id in range(NUM_CLASSES):

        true_positive = int(cm[class_id, class_id])

        false_positive = int(
            cm[:, class_id].sum()
            - true_positive
        )

        false_negative = int(
            cm[class_id, :].sum()
            - true_positive
        )

        support = int(
            cm[class_id, :].sum()
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        union = (
            true_positive
            + false_positive
            + false_negative
        )

        dice_denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        iou = (
            true_positive / union
            if union > 0
            else 0.0
        )

        dice = (
            2 * true_positive / dice_denominator
            if dice_denominator > 0
            else 0.0
        )

        class_name = EVAL_CLASS_NAMES[class_id]

        per_class[class_name] = {
            "class_id": class_id,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "iou": float(iou),
            "dice": float(dice),
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        iou_values.append(iou)
        dice_values.append(dice)

    metrics = {
        "overall": {
            "accuracy": float(accuracy),
            "macro_precision": float(
                np.mean(precision_values)
            ),
            "macro_recall": float(
                np.mean(recall_values)
            ),
            "macro_f1": float(
                np.mean(f1_values)
            ),
            "macro_iou": float(
                np.mean(iou_values)
            ),
            "macro_dice": float(
                np.mean(dice_values)
            ),
            "total_pixels": total_pixels,
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ============================================================
# SIMPAN JSON
# ============================================================

def save_json(
    metrics: dict,
    output_path: Path,
) -> None:

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


# ============================================================
# SIMPAN CSV METRIK
# ============================================================

def save_metrics_csv(
    metrics: dict,
    output_path: Path,
) -> None:

    rows = []

    overall = metrics["overall"]

    rows.append(
        {
            "class": "overall",
            "class_id": "",
            "precision": overall["macro_precision"],
            "recall": overall["macro_recall"],
            "f1_score": overall["macro_f1"],
            "iou": overall["macro_iou"],
            "dice": overall["macro_dice"],
            "support": overall["total_pixels"],
        }
    )

    for class_name, values in metrics["per_class"].items():

        rows.append(
            {
                "class": class_name,
                "class_id": values["class_id"],
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1_score"],
                "iou": values["iou"],
                "dice": values["dice"],
                "support": values["support"],
            }
        )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "class_id",
                "precision",
                "recall",
                "f1_score",
                "iou",
                "dice",
                "support",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SIMPAN CONFUSION MATRIX CSV
# ============================================================

def save_confusion_matrix_csv(
    cm: np.ndarray,
    output_path: Path,
) -> None:

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["actual/predicted"] + EVAL_CLASS_NAMES
        )

        for index, row in enumerate(cm):

            writer.writerow(
                [EVAL_CLASS_NAMES[index]]
                + row.tolist()
            )


# ============================================================
# SIMPAN CONFUSION MATRIX PNG
# ============================================================

def save_confusion_matrix_png(
    cm: np.ndarray,
    output_path: Path,
) -> None:

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        cm,
        interpolation="nearest",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=EVAL_CLASS_NAMES,
        yticklabels=EVAL_CLASS_NAMES,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix - UPerNet",
    )

    threshold = (
        cm.max() / 2.0
        if cm.max() > 0
        else 0.0
    )

    for row in range(NUM_CLASSES):

        for column in range(NUM_CLASSES):

            axis.text(
                column,
                row,
                format(cm[row, column], "d"),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, column] > threshold
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("EVALUASI UPERNET - SELURUH HASIL PREDIKSI")
    print("=" * 80)

    print(f"Folder prediksi : {EVAL_PRED_DIR}")
    print(f"Folder GT       : {EVAL_GT_DIR}")
    print(f"Output evaluasi : {EVAL_OUTPUT_DIR}")
    print()

    if not EVAL_GT_DIR.exists():
        raise FileNotFoundError(
            "Folder ground-truth tidak ditemukan:\n"
            f"{EVAL_GT_DIR}\n\n"
            "Silakan sesuaikan EVAL_GT_DIR pada config.py."
        )

    prediction_files = get_prediction_files(
        EVAL_PRED_DIR
    )

    print(
        f"Jumlah file prediksi: "
        f"{len(prediction_files)}"
    )
    print()

    global_cm = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    evaluated_files = []
    skipped_files = []

    for prediction_path in tqdm(
        prediction_files,
        desc="Evaluasi UPerNet",
        unit="file",
    ):

        ground_truth_path = find_ground_truth(
            prediction_path
        )

        if ground_truth_path is None:

            skipped_files.append(
                {
                    "prediction": prediction_path.name,
                    "reason": "Ground-truth tidak ditemukan",
                }
            )

            print(
                f"\nGT tidak ditemukan untuk: "
                f"{prediction_path.name}"
            )

            continue

        try:

            prediction_mask, prediction_meta = read_mask(
                prediction_path
            )

            ground_truth_mask, ground_truth_meta = read_mask(
                ground_truth_path
            )

            # Validasi ukuran
            if prediction_mask.shape != ground_truth_mask.shape:

                skipped_files.append(
                    {
                        "prediction": prediction_path.name,
                        "ground_truth": ground_truth_path.name,
                        "reason": (
                            "Ukuran prediksi dan GT berbeda: "
                            f"{prediction_mask.shape} vs "
                            f"{ground_truth_mask.shape}"
                        ),
                    }
                )

                print(
                    f"\nUkuran berbeda untuk "
                    f"{prediction_path.name}: "
                    f"{prediction_mask.shape} vs "
                    f"{ground_truth_mask.shape}"
                )

                continue

            prediction_mask = normalize_mask(
                prediction_mask
            )

            ground_truth_mask = normalize_mask(
                ground_truth_mask
            )

            update_confusion_matrix(
                global_cm,
                ground_truth_mask.flatten(),
                prediction_mask.flatten(),
            )

            evaluated_files.append(
                {
                    "prediction": prediction_path.name,
                    "ground_truth": ground_truth_path.name,
                    "shape": list(prediction_mask.shape),
                }
            )

        except Exception as error:

            skipped_files.append(
                {
                    "prediction": prediction_path.name,
                    "ground_truth": ground_truth_path.name,
                    "reason": str(error),
                }
            )

            print(
                f"\nGagal mengevaluasi "
                f"{prediction_path.name}: {error}"
            )

    if len(evaluated_files) == 0:
        raise RuntimeError(
            "Tidak ada file yang berhasil dievaluasi.\n"
            "Periksa lokasi dan nama ground-truth mask."
        )

    metrics = calculate_metrics(
        global_cm
    )

    metrics["files"] = {
        "total_prediction_files": len(prediction_files),
        "evaluated_files": len(evaluated_files),
        "skipped_files": len(skipped_files),
        "evaluated_details": evaluated_files,
        "skipped_details": skipped_files,
    }

    # ========================================================
    # SIMPAN HASIL
    # ========================================================

    save_json(
        metrics,
        EVAL_METRICS_JSON,
    )

    save_metrics_csv(
        metrics,
        EVAL_METRICS_CSV,
    )

    save_confusion_matrix_csv(
        global_cm,
        EVAL_CONFUSION_MATRIX_CSV,
    )

    save_confusion_matrix_png(
        global_cm,
        EVAL_CONFUSION_MATRIX_PNG,
    )

    # ========================================================
    # CETAK HASIL
    # ========================================================

    overall = metrics["overall"]

    print()
    print("=" * 80)
    print("HASIL EVALUASI UPERNET")
    print("=" * 80)

    print(
        f"File dievaluasi : "
        f"{len(evaluated_files)}"
    )

    print(
        f"File dilewati   : "
        f"{len(skipped_files)}"
    )

    print()

    print(
        f"Accuracy        : "
        f"{overall['accuracy']:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{overall['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{overall['macro_recall']:.6f}"
    )

    print(
        f"Macro F1-score  : "
        f"{overall['macro_f1']:.6f}"
    )

    print(
        f"Macro IoU       : "
        f"{overall['macro_iou']:.6f}"
    )

    print(
        f"Macro Dice      : "
        f"{overall['macro_dice']:.6f}"
    )

    print()
    print("METRIK PER KELAS")

    for class_name, values in metrics["per_class"].items():

        print()
        print(f"Kelas: {class_name}")

        print(
            f"  Precision : "
            f"{values['precision']:.6f}"
        )

        print(
            f"  Recall    : "
            f"{values['recall']:.6f}"
        )

        print(
            f"  F1-score  : "
            f"{values['f1_score']:.6f}"
        )

        print(
            f"  IoU       : "
            f"{values['iou']:.6f}"
        )

        print(
            f"  Dice      : "
            f"{values['dice']:.6f}"
        )

        print(
            f"  Support   : "
            f"{values['support']}"
        )

    print()
    print("CONFUSION MATRIX")
    print(global_cm)

    print()
    print("FILE HASIL EVALUASI")
    print(f"JSON  : {EVAL_METRICS_JSON}")
    print(f"CSV   : {EVAL_METRICS_CSV}")
    print(f"CM CSV: {EVAL_CONFUSION_MATRIX_CSV}")
    print(f"CM PNG: {EVAL_CONFUSION_MATRIX_PNG}")

    print("=" * 80)


if __name__ == "__main__":
    main()