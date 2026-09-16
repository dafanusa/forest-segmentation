from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Mask hasil prediksi ResNet50
PREDICTION_MASK_PATH = (
    BASE_DIR
    / "outputs"
    / "Danum_resnet50_mask.tif"
)

# Mask ground truth / label referensi
GROUND_TRUTH_MASK_PATH = (
    BASE_DIR
    / "dataset"
    / "data"
    / "input"
    / "Dan_2014_RGB_project_to_CHM.tif"
)

# Folder output evaluasi
OUTPUT_DIR = BASE_DIR / "outputs"

# File hasil metrik
METRICS_CSV_PATH = (
    OUTPUT_DIR
    / "metrics_resnet50.csv"
)

# Ukuran window evaluasi
EVAL_TILE_SIZE = 1024

# Nilai label
BACKGROUND_CLASS = 0
TREE_CLASS = 1

# Jika ground truth memiliki nilai NoData,
# nilai tersebut akan diabaikan.
IGNORE_NODATA = True


# ============================================================
# VALIDASI FILE
# ============================================================

def validate_files():

    if not PREDICTION_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Mask prediksi tidak ditemukan:\n"
            f"{PREDICTION_MASK_PATH}"
        )

    if not GROUND_TRUTH_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Mask ground truth tidak ditemukan:\n"
            f"{GROUND_TRUTH_MASK_PATH}"
        )


# ============================================================
# VALIDASI RASTER
# ============================================================

def validate_rasters(pred_src, gt_src):

    print("\n========== VALIDASI RASTER ==========")

    print(
        f"Ukuran prediksi    : "
        f"{pred_src.width} x {pred_src.height}"
    )

    print(
        f"Ukuran ground truth: "
        f"{gt_src.width} x {gt_src.height}"
    )

    print(f"CRS prediksi       : {pred_src.crs}")
    print(f"CRS ground truth   : {gt_src.crs}")

    if pred_src.width != gt_src.width:
        raise ValueError(
            "Lebar mask prediksi dan ground truth berbeda."
        )

    if pred_src.height != gt_src.height:
        raise ValueError(
            "Tinggi mask prediksi dan ground truth berbeda."
        )

    if pred_src.count < 1:
        raise ValueError(
            "Mask prediksi tidak memiliki band."
        )

    if gt_src.count < 1:
        raise ValueError(
            "Mask ground truth tidak memiliki band."
        )

    if pred_src.transform != gt_src.transform:
        print(
            "[WARNING] Transform raster berbeda."
        )

    if pred_src.crs != gt_src.crs:
        print(
            "[WARNING] CRS raster berbeda."
        )

    print("=====================================\n")


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion_matrix(
    prediction,
    ground_truth,
    nodata_value
):

    # Validasi kelas prediksi dan ground truth.
    # Semua nilai > 0 dianggap sebagai tree.
    prediction_binary = (
        prediction > 0
    ).astype(np.uint8)

    ground_truth_binary = (
        ground_truth > 0
    ).astype(np.uint8)

    # Abaikan NoData ground truth jika tersedia.
    valid = np.ones(
        ground_truth.shape,
        dtype=bool
    )

    if IGNORE_NODATA and nodata_value is not None:

        valid = (
            ground_truth != nodata_value
        )

    prediction_binary = prediction_binary[valid]
    ground_truth_binary = ground_truth_binary[valid]

    tp = np.sum(
        (prediction_binary == TREE_CLASS)
        &
        (ground_truth_binary == TREE_CLASS)
    )

    tn = np.sum(
        (prediction_binary == BACKGROUND_CLASS)
        &
        (ground_truth_binary == BACKGROUND_CLASS)
    )

    fp = np.sum(
        (prediction_binary == TREE_CLASS)
        &
        (ground_truth_binary == BACKGROUND_CLASS)
    )

    fn = np.sum(
        (prediction_binary == BACKGROUND_CLASS)
        &
        (ground_truth_binary == TREE_CLASS)
    )

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn),
        int(valid.sum())
    )


# ============================================================
# HITUNG METRIK
# ============================================================

def calculate_metrics(tp, tn, fp, fn):

    total = tp + tn + fp + fn

    # Pixel Accuracy
    pixel_accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    # Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    # IoU kelas tree
    iou_tree = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0 else 0.0
    )

    # IoU background
    iou_background = (
        tn / (tn + fp + fn)
        if (tn + fp + fn) > 0 else 0.0
    )

    # Mean IoU
    mean_iou = (
        (iou_tree + iou_background) / 2
    )

    # Dice / F1
    dice_tree = (
        (2 * tp) / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0 else 0.0
    )

    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    return {
        "total_pixels": total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "pixel_accuracy": pixel_accuracy,
        "precision_tree": precision,
        "recall_tree": recall,
        "specificity_background": specificity,
        "iou_tree": iou_tree,
        "iou_background": iou_background,
        "mean_iou": mean_iou,
        "dice_tree": dice_tree,
        "f1_tree": dice_tree
    }


# ============================================================
# EVALUASI RASTER
# ============================================================

def evaluate_raster():

    validate_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_valid = 0

    print("=" * 65)
    print("EVALUASI SEGMENTASI RESNET50")
    print("=" * 65)

    print(f"Prediksi    : {PREDICTION_MASK_PATH}")
    print(f"Ground truth: {GROUND_TRUTH_MASK_PATH}")
    print(f"Tile evaluasi: {EVAL_TILE_SIZE}")

    with rasterio.open(
        PREDICTION_MASK_PATH
    ) as pred_src, rasterio.open(
        GROUND_TRUTH_MASK_PATH
    ) as gt_src:

        validate_rasters(
            pred_src,
            gt_src
        )

        width = pred_src.width
        height = pred_src.height

        pred_nodata = pred_src.nodata
        gt_nodata = gt_src.nodata

        print(f"NoData prediksi    : {pred_nodata}")
        print(f"NoData ground truth: {gt_nodata}")

        # Ground truth biasanya memakai NoData.
        # Jika tidak ada, gunakan None.
        nodata_value = gt_nodata

        total_tiles = (
            ((height - 1) // EVAL_TILE_SIZE + 1)
            *
            ((width - 1) // EVAL_TILE_SIZE + 1)
        )

        tile_number = 0

        for top in range(
            0,
            height,
            EVAL_TILE_SIZE
        ):

            for left in range(
                0,
                width,
                EVAL_TILE_SIZE
            ):

                bottom = min(
                    top + EVAL_TILE_SIZE,
                    height
                )

                right = min(
                    left + EVAL_TILE_SIZE,
                    width
                )

                tile_height = bottom - top
                tile_width = right - left

                window = Window(
                    col_off=left,
                    row_off=top,
                    width=tile_width,
                    height=tile_height
                )

                prediction = pred_src.read(
                    1,
                    window=window
                )

                ground_truth = gt_src.read(
                    1,
                    window=window
                )

                tp, tn, fp, fn, valid_count = (
                    update_confusion_matrix(
                        prediction,
                        ground_truth,
                        nodata_value
                    )
                )

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_valid += valid_count

                tile_number += 1

                print(
                    f"[{tile_number}/{total_tiles}] "
                    f"Evaluasi tile selesai"
                )

    metrics = calculate_metrics(
        total_tp,
        total_tn,
        total_fp,
        total_fn
    )

    # ========================================================
    # DISTRIBUSI KELAS
    # ========================================================

    ground_truth_tree = (
        total_tp + total_fn
    )

    ground_truth_background = (
        total_tn + total_fp
    )

    prediction_tree = (
        total_tp + total_fp
    )

    prediction_background = (
        total_tn + total_fn
    )

    gt_tree_percentage = (
        ground_truth_tree / total_valid * 100
        if total_valid > 0 else 0.0
    )

    gt_background_percentage = (
        ground_truth_background / total_valid * 100
        if total_valid > 0 else 0.0
    )

    pred_tree_percentage = (
        prediction_tree / total_valid * 100
        if total_valid > 0 else 0.0
    )

    pred_background_percentage = (
        prediction_background / total_valid * 100
        if total_valid > 0 else 0.0
    )

    # Tambahkan distribusi ke dictionary
    metrics.update({

        "ground_truth_tree_pixels": ground_truth_tree,
        "ground_truth_background_pixels": (
            ground_truth_background
        ),

        "prediction_tree_pixels": prediction_tree,
        "prediction_background_pixels": (
            prediction_background
        ),

        "ground_truth_tree_percentage": (
            gt_tree_percentage
        ),

        "ground_truth_background_percentage": (
            gt_background_percentage
        ),

        "prediction_tree_percentage": (
            pred_tree_percentage
        ),

        "prediction_background_percentage": (
            pred_background_percentage
        )
    })

    # ========================================================
    # CETAK HASIL
    # ========================================================

    print("\n" + "=" * 65)
    print("HASIL EVALUASI RESNET50")
    print("=" * 65)

    print(f"Total piksel valid : {metrics['total_pixels']:,}")

    print("\n--- CONFUSION MATRIX ---")

    print(f"True Positive  (TP): {metrics['true_positive']:,}")
    print(f"True Negative  (TN): {metrics['true_negative']:,}")
    print(f"False Positive (FP): {metrics['false_positive']:,}")
    print(f"False Negative (FN): {metrics['false_negative']:,}")

    print("\n--- METRIK SEGMENTASI ---")

    print(
        f"Pixel Accuracy       : "
        f"{metrics['pixel_accuracy'] * 100:.4f}%"
    )

    print(
        f"Precision Tree       : "
        f"{metrics['precision_tree'] * 100:.4f}%"
    )

    print(
        f"Recall Tree          : "
        f"{metrics['recall_tree'] * 100:.4f}%"
    )

    print(
        f"Specificity Background: "
        f"{metrics['specificity_background'] * 100:.4f}%"
    )

    print(
        f"IoU Background       : "
        f"{metrics['iou_background'] * 100:.4f}%"
    )

    print(
        f"IoU Tree             : "
        f"{metrics['iou_tree'] * 100:.4f}%"
    )

    print(
        f"Mean IoU             : "
        f"{metrics['mean_iou'] * 100:.4f}%"
    )

    print(
        f"Dice / F1 Tree       : "
        f"{metrics['dice_tree'] * 100:.4f}%"
    )

    print("\n--- DISTRIBUSI KELAS ---")

    print(
        f"Ground Truth Tree       : "
        f"{metrics['ground_truth_tree_percentage']:.4f}%"
    )

    print(
        f"Ground Truth Background : "
        f"{metrics['ground_truth_background_percentage']:.4f}%"
    )

    print(
        f"Prediction Tree         : "
        f"{metrics['prediction_tree_percentage']:.4f}%"
    )

    print(
        f"Prediction Background   : "
        f"{metrics['prediction_background_percentage']:.4f}%"
    )

    # ========================================================
    # SIMPAN CSV
    # ========================================================

    with open(
        METRICS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "metric",
            "value"
        ])

        for key, value in metrics.items():

            writer.writerow([
                key,
                value
            ])

    print("\n======================================")
    print("EVALUASI SELESAI")
    print("======================================")

    print(
        f"CSV hasil evaluasi:\n"
        f"{METRICS_CSV_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_raster()