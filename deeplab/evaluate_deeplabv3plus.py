from __future__ import annotations

from pathlib import Path
import sys
import json
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
)


# ============================================================
# IMPORT CONFIG
# ============================================================

try:
    from .config import (
        NUM_CLASSES,
        ID2LABEL,
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

    from deeplab.config import (
        NUM_CLASSES,
        ID2LABEL,
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
# KONFIGURASI EVALUASI
# ============================================================

PREDICTION_SUFFIX = "_pred"

# Kandidat nama ground-truth:
# 100_mask.tif
# 100.tif
GROUND_TRUTH_SUFFIX = "_mask"

IGNORE_INDEX = 255


# ============================================================
# GET TIFF FILES
# ============================================================

def get_prediction_files(
    prediction_dir: Path,
) -> list[Path]:
    """
    Mengambil seluruh file hasil prediksi.
    """

    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Folder prediksi tidak ditemukan:\n"
            f"{prediction_dir}"
        )

    files = sorted(
        list(prediction_dir.glob("*.tif"))
        + list(prediction_dir.glob("*.tiff"))
    )

    files = [
        file
        for file in files
        if PREDICTION_SUFFIX in file.stem
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"Tidak ditemukan file prediksi di:\n"
            f"{prediction_dir}"
        )

    return files


# ============================================================
# CARI GROUND-TRUTH MASK
# ============================================================

def find_ground_truth(
    prediction_path: Path,
) -> Path | None:
    """
    Mencari ground-truth berdasarkan nama file prediksi.

    Contoh:
        100_pred.tif
    akan mencari:
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
# BACA RASTER MASK
# ============================================================

def read_mask(
    mask_path: Path,
) -> tuple[np.ndarray, dict]:
    """
    Membaca mask raster sebagai array 2D.
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
    Menormalisasi mask menjadi label kelas:
        0 = background
        1 = tree

    Mendukung mask dengan nilai:
        0 dan 1
        0 dan 255
    """

    mask = mask.astype(np.int64)

    unique_values = np.unique(mask)

    # Jika mask biner 0 dan 255, ubah 255 menjadi 1
    if 255 in unique_values and np.max(unique_values) <= 255:
        mask = np.where(mask == 255, 1, mask)

    # Untuk mask biner, semua nilai positif dianggap tree
    if NUM_CLASSES == 2:
        mask = np.where(mask > 0, 1, 0)

    return mask


# ============================================================
# HITUNG CONFUSION MATRIX GLOBAL
# ============================================================

def update_confusion_matrix(
    global_cm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Menambahkan confusion matrix dari satu citra
    ke confusion matrix global.
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
# HITUNG METRIK DARI CONFUSION MATRIX
# ============================================================

def calculate_metrics(
    cm: np.ndarray,
) -> dict:
    """
    Menghitung metrik dari confusion matrix global.
    """

    total = cm.sum()

    accuracy = (
        np.trace(cm) / total
        if total > 0
        else 0.0
    )

    per_class = {}

    ious = []
    dices = []
    precisions = []
    recalls = []
    f1_scores = []

    for class_id in range(NUM_CLASSES):

        true_positive = cm[class_id, class_id]

        false_positive = (
            cm[:, class_id].sum()
            - true_positive
        )

        false_negative = (
            cm[class_id, :].sum()
            - true_positive
        )

        support = cm[class_id, :].sum()

        union = (
            true_positive
            + false_positive
            + false_negative
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        iou = (
            true_positive / union
            if union > 0
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

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        class_name = EVAL_CLASS_NAMES[class_id]

        per_class[class_name] = {
            "class_id": class_id,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "iou": float(iou),
            "dice": float(dice),
            "support": int(support),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
        }

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    metrics = {
        "overall": {
            "accuracy": float(accuracy),
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_scores)),
            "macro_iou": float(np.mean(ious)),
            "macro_dice": float(np.mean(dices)),
            "total_pixels": int(total),
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

    class_names = EVAL_CLASS_NAMES

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["actual/predicted"] + class_names
        )

        for index, row in enumerate(cm):

            writer.writerow(
                [class_names[index]] + row.tolist()
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

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=EVAL_CLASS_NAMES,
        yticklabels=EVAL_CLASS_NAMES,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix - DeepLab",
    )

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0

    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):

            axis.text(
                column,
                row,
                format(cm[row, column], "d"),
                ha="center",
                va="center",
                color="white"
                if cm[row, column] > threshold
                else "black",
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print("=" * 80)
    print("EVALUASI DEEPLAB - SELURUH HASIL PREDIKSI")
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
        f"Jumlah file prediksi ditemukan: "
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
        desc="Evaluasi DeepLab",
        unit="file",
    ):

        gt_path = find_ground_truth(
            prediction_path
        )

        if gt_path is None:

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
                gt_path
            )

            if prediction_mask.shape != ground_truth_mask.shape:

                skipped_files.append(
                    {
                        "prediction": prediction_path.name,
                        "ground_truth": gt_path.name,
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
                    "ground_truth": gt_path.name,
                    "shape": list(prediction_mask.shape),
                }
            )

        except Exception as error:

            skipped_files.append(
                {
                    "prediction": prediction_path.name,
                    "ground_truth": gt_path.name,
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

    # Simpan seluruh hasil
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
    print("HASIL EVALUASI DEEPLAB")
    print("=" * 80)

    print(f"File dievaluasi : {len(evaluated_files)}")
    print(f"File dilewati   : {len(skipped_files)}")
    print()

    print(f"Accuracy        : {overall['accuracy']:.6f}")
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
            f"  Precision : {values['precision']:.6f}"
        )
        print(
            f"  Recall    : {values['recall']:.6f}"
        )
        print(
            f"  F1-score  : {values['f1_score']:.6f}"
        )
        print(
            f"  IoU       : {values['iou']:.6f}"
        )
        print(
            f"  Dice      : {values['dice']:.6f}"
        )
        print(
            f"  Support   : {values['support']}"
        )

    print()
    print("CONFUSION MATRIX")
    print(global_cm)

    print()
    print("FILE HASIL EVALUASI")
    print(f"JSON : {EVAL_METRICS_JSON}")
    print(f"CSV  : {EVAL_METRICS_CSV}")
    print(f"CM CSV: {EVAL_CONFUSION_MATRIX_CSV}")
    print(f"CM PNG: {EVAL_CONFUSION_MATRIX_PNG}")

    print("=" * 80)


if __name__ == "__main__":
    main()