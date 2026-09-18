from pathlib import Path
import csv
import gc
import json

import cv2
import numpy as np
import rasterio
import torch

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from config import (
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    TREE_THRESHOLD,
    MIN_INSTANCE_AREA,
    WATERSHED_MIN_DISTANCE,
    DEVICE,
    SEMANTIC_OUTPUT_DIR,
    INSTANCE_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR,
    create_directories,
)


# ============================================================
# KONFIGURASI FOLDER INPUT
# ============================================================

INPUT_TIF_DIR = Path(
    r"D:\MAHASISWA\SEMESTER 5\FUNGSIONAL\PRAKTIKUM\STRD-Net\tcd_dataset\dataset"
)

DETAIL_OUTPUT_DIR = Path("outputs/detail")
STATISTICS_OUTPUT_DIR = Path("outputs/statistics")

MODEL_PATH = CHECKPOINT_DIR / "best_model"

# Folder khusus dua jenis overlay
SEMANTIC_OVERLAY_DIR = Path("outputs/overlay_semantic")
BBOX_OVERLAY_DIR = Path("outputs/overlay_bbox")


# ============================================================
# MEMBUAT FOLDER OUTPUT
# ============================================================

def create_all_directories():

    create_directories()

    SEMANTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INSTANCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OVERLAY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DETAIL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEMANTIC_OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BBOX_OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MEMBACA CITRA TIFF
# ============================================================

def read_tif(path):

    with rasterio.open(path) as src:
        image = src.read()

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    if image.shape[2] == 1:

        image = np.repeat(
            image,
            3,
            axis=2,
        )

    if image.shape[2] == 2:

        image = np.concatenate(
            [
                image,
                image[:, :, 1:2],
            ],
            axis=2,
        )

    if image.shape[2] > 3:

        image = image[:, :, :3]

    image = image.astype(np.float32)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            (image - image_min)
            / (image_max - image_min)
            * 255.0
        )

    return image.astype(np.uint8)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Memuat processor...")

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_PATH
    )

    print("Memuat model...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Model berhasil dimuat pada device: {DEVICE}"
    )

    return processor, model


# ============================================================
# PREDIKSI SATU PATCH
# ============================================================

def predict_patch(
    image,
    processor,
    model,
):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(
                image.shape[0],
                image.shape[1],
            ),
            mode="bilinear",
            align_corners=False,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        tree_probability = probabilities[:, 1]

        semantic = (
            tree_probability > TREE_THRESHOLD
        ).cpu().numpy()[0].astype(np.uint8)

    return semantic


# ============================================================
# PREDIKSI SLIDING WINDOW
# ============================================================

def predict_sliding_window(
    image,
    processor,
    model,
    patch_size=512,
    stride=384,
):

    height, width = image.shape[:2]

    prediction_sum = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    prediction_count = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    total_patch = (
        ((height - 1) // stride) + 1
    ) * (
        ((width - 1) // stride) + 1
    )

    current_patch = 0

    for y in range(
        0,
        height,
        stride,
    ):

        for x in range(
            0,
            width,
            stride,
        ):

            current_patch += 1

            y2 = min(
                y + patch_size,
                height,
            )

            x2 = min(
                x + patch_size,
                width,
            )

            patch = image[
                y:y2,
                x:x2,
            ]

            ph, pw = patch.shape[:2]

            if (
                ph != patch_size
                or pw != patch_size
            ):

                padded = np.zeros(
                    (
                        patch_size,
                        patch_size,
                        3,
                    ),
                    dtype=np.uint8,
                )

                padded[
                    :ph,
                    :pw,
                ] = patch

                patch_input = padded

            else:

                patch_input = patch

            semantic_patch = predict_patch(
                patch_input,
                processor,
                model,
            )

            semantic_patch = semantic_patch[
                :ph,
                :pw,
            ]

            prediction_sum[
                y:y2,
                x:x2,
            ] += semantic_patch

            prediction_count[
                y:y2,
                x:x2,
            ] += 1

            if (
                current_patch % 20 == 0
                or current_patch == total_patch
            ):

                print(
                    f"  Patch {current_patch}/{total_patch}",
                    end="\r",
                )

    semantic_mask = (
        prediction_sum
        / np.maximum(
            prediction_count,
            1,
        )
        > 0.5
    ).astype(np.uint8)

    print()

    return semantic_mask


# ============================================================
# SEMANTIC MASK MENJADI INSTANCE MASK
# ============================================================

def semantic_to_instance(semantic_mask):

    binary_mask = (
        semantic_mask > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    cleaned_mask = np.zeros_like(
        binary_mask
    )

    for label_id in range(
        1,
        num_labels,
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= MIN_INSTANCE_AREA:

            cleaned_mask[
                labels == label_id
            ] = 1

    if cleaned_mask.sum() == 0:

        return np.zeros_like(
            cleaned_mask,
            dtype=np.int32,
        )

    distance = cv2.distanceTransform(
        cleaned_mask,
        cv2.DIST_L2,
        5,
    )

    coordinates = peak_local_max(
        distance,
        min_distance=WATERSHED_MIN_DISTANCE,
        threshold_abs=2,
        labels=cleaned_mask,
    )

    markers = np.zeros_like(
        cleaned_mask,
        dtype=np.int32,
    )

    for idx, (
        row,
        col,
    ) in enumerate(
        coordinates,
        start=1,
    ):

        markers[
            row,
            col,
        ] = idx

    if len(coordinates) == 0:

        markers[
            cleaned_mask > 0
        ] = 1

    else:

        markers = ndimage.label(
            markers > 0
        )[0]

    instance_mask = watershed(
        -distance,
        markers,
        mask=cleaned_mask,
    )

    return instance_mask.astype(
        np.int32
    )


# ============================================================
# MENGAMBIL DETAIL SETIAP INSTANCE
# ============================================================

def extract_instance_details(
    instance_mask,
):

    instance_details = []

    unique_instances = np.unique(
        instance_mask
    )

    for instance_id in unique_instances:

        if instance_id == 0:
            continue

        binary = (
            instance_mask == instance_id
        ).astype(np.uint8)

        area_pixels = int(
            np.sum(binary)
        )

        if area_pixels <= 0:
            continue

        x, y, width, height = cv2.boundingRect(
            binary
        )

        moments = cv2.moments(
            binary
        )

        if moments["m00"] != 0:

            centroid_x = (
                moments["m10"]
                / moments["m00"]
            )

            centroid_y = (
                moments["m01"]
                / moments["m00"]
            )

        else:

            centroid_x = (
                x + width / 2
            )

            centroid_y = (
                y + height / 2
            )

        detail = {
            "instance_id": int(
                instance_id
            ),
            "tree_type": (
                "Tree - jenis belum teridentifikasi"
            ),
            "area_pixels": area_pixels,
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_width": int(width),
            "bbox_height": int(height),
            "bbox_x2": int(
                x + width - 1
            ),
            "bbox_y2": int(
                y + height - 1
            ),
            "centroid_x": round(
                float(centroid_x),
                2,
            ),
            "centroid_y": round(
                float(centroid_y),
                2,
            ),
        }

        instance_details.append(
            detail
        )

    return instance_details


# ============================================================
# FUNGSI TEKS DENGAN BACKGROUND
# ============================================================

def draw_text_with_background(
    image,
    text,
    position,
    font_scale=0.42,
    text_color=(255, 255, 255),
    background_color=(0, 0, 0),
    thickness=1,
    padding=4,
):

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    rect_x1 = max(
        x - padding,
        0,
    )

    rect_y1 = max(
        y - text_height - padding,
        0,
    )

    rect_x2 = min(
        x + text_width + padding,
        image.shape[1] - 1,
    )

    rect_y2 = min(
        y + baseline + padding,
        image.shape[0] - 1,
    )

    cv2.rectangle(
        image,
        (
            rect_x1,
            rect_y1,
        ),
        (
            rect_x2,
            rect_y2,
        ),
        background_color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            x,
            y,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# OVERLAY 1: HANYA AREA POHON DAN PERSENTASE
# ============================================================

def create_semantic_overlay(
    image,
    semantic_mask,
    statistics,
):

    height, width = semantic_mask.shape

    # Background hitam
    overlay = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # Area pohon hijau
    overlay[
        semantic_mask > 0
    ] = [
        0,
        255,
        0,
    ]

    # Panel informasi
    panel_lines = [
        "SEMANTIC TREE DETECTION",
        (
            f"Ukuran: "
            f"{statistics['width']} x "
            f"{statistics['height']} px"
        ),
        (
            f"Tree: "
            f"{statistics['tree_percentage']:.2f}%"
        ),
        (
            f"Background: "
            f"{statistics['background_percentage']:.2f}%"
        ),
        (
            f"Piksel pohon: "
            f"{statistics['tree_pixels']:,}"
        ),
        (
            f"Piksel background: "
            f"{statistics['background_pixels']:,}"
        ),
    ]

    panel_x = 25
    panel_y = 35

    for line_index, line in enumerate(
        panel_lines
    ):

        draw_text_with_background(
            overlay,
            line,
            (
                panel_x,
                panel_y + line_index * 25,
            ),
            font_scale=(
                0.55
                if line_index == 0
                else 0.45
            ),
            text_color=(
                255,
                255,
                255,
            ),
            background_color=(
                0,
                0,
                0,
            ),
            padding=3,
        )

    return overlay


# ============================================================
# OVERLAY 2:
# CITRA ASLI + AREA HIJAU TRANSPARAN +
# BOUNDARY MERAH + BOUNDING BOX MERAH +
# CENTROID DI TENGAH POHON
# ============================================================

def create_bbox_overlay(
    image,
    semantic_mask,
    instance_mask,
):

    # Pertahankan citra asli sebagai background
    overlay = image.copy()

    # --------------------------------------------------------
    # Area pohon diberi warna hijau transparan
    # --------------------------------------------------------

    tree_area = semantic_mask > 0

    green_color = np.array(
        [
            0,
            255,
            0,
        ],
        dtype=np.float32,
    )

    # Tingkat transparansi warna hijau
    alpha = 0.35

    overlay[tree_area] = (
        (
            1 - alpha
        )
        * overlay[tree_area].astype(np.float32)
        + alpha * green_color
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Boundary, bounding box, dan centroid
    # --------------------------------------------------------

    unique_instances = np.unique(
        instance_mask
    )

    for instance_id in unique_instances:

        # Abaikan background
        if instance_id == 0:
            continue

        # Mask untuk satu pohon
        binary = (
            instance_mask == instance_id
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, width, height = cv2.boundingRect(
            binary
        )

        x2 = x + width - 1
        y2 = y + height - 1

        # Warna merah dalam format RGB
        red_color = (
            255,
            0,
            0,
        )

        # ----------------------------------------------------
        # Bounding box merah
        # ----------------------------------------------------

        cv2.rectangle(
            overlay,
            (
                x,
                y,
            ),
            (
                x2,
                y2,
            ),
            red_color,
            2,
        )

        # ----------------------------------------------------
        # Boundary/kontur merah
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            red_color,
            2,
        )

        # ----------------------------------------------------
        # Menghitung centroid tengah pohon
        # ----------------------------------------------------

        moments = cv2.moments(
            binary
        )

        if moments["m00"] != 0:

            centroid_x = int(
                round(
                    moments["m10"]
                    / moments["m00"]
                )
            )

            centroid_y = int(
                round(
                    moments["m01"]
                    / moments["m00"]
                )
            )

        else:

            centroid_x = int(
                round(
                    x + width / 2
                )
            )

            centroid_y = int(
                round(
                    y + height / 2
                )
            )

        # Pastikan centroid berada di dalam ukuran citra
        centroid_x = max(
            0,
            min(
                centroid_x,
                overlay.shape[1] - 1,
            ),
        )

        centroid_y = max(
            0,
            min(
                centroid_y,
                overlay.shape[0] - 1,
            ),
        )

        # ----------------------------------------------------
        # Titik centroid
        # Kuning sebagai penanda titik tengah pohon
        # dengan outline merah agar tetap terlihat
        # ----------------------------------------------------

        cv2.circle(
            overlay,
            (
                centroid_x,
                centroid_y,
            ),
            7,
            (
                255,
                0,
                0,
            ),
            -1,
        )

        cv2.circle(
            overlay,
            (
                centroid_x,
                centroid_y,
            ),
            4,
            (
                255,
                255,
                0,
            ),
            -1,
        )

    return overlay


# ============================================================
# MEMBUAT PREVIEW SEMANTIC BERWARNA
# ============================================================

def create_semantic_preview(
    semantic_mask
):

    preview = np.zeros(
        (
            semantic_mask.shape[0],
            semantic_mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    preview[
        semantic_mask == 0
    ] = [
        0,
        0,
        0,
    ]

    preview[
        semantic_mask > 0
    ] = [
        0,
        255,
        0,
    ]

    return preview


# ============================================================
# MEMBUAT PREVIEW INSTANCE BERWARNA
# ============================================================

def create_instance_preview(
    instance_mask
):

    height, width = instance_mask.shape

    preview = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    unique_instances = np.unique(
        instance_mask
    )

    rng = np.random.default_rng(42)

    for instance_id in unique_instances:

        if instance_id == 0:
            continue

        color = rng.integers(
            low=50,
            high=255,
            size=3,
            dtype=np.uint8,
        )

        preview[
            instance_mask == instance_id
        ] = color

    return preview


# ============================================================
# MENGHITUNG STATISTIK PREDIKSI
# ============================================================

def calculate_statistics(
    image,
    semantic_mask,
    instance_mask,
):

    height, width = semantic_mask.shape

    total_pixels = height * width

    tree_pixels = int(
        np.sum(semantic_mask > 0)
    )

    background_pixels = (
        total_pixels - tree_pixels
    )

    tree_percentage = (
        tree_pixels / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    background_percentage = (
        background_pixels / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    instance_ids = np.unique(
        instance_mask
    )

    instance_ids = instance_ids[
        instance_ids != 0
    ]

    instance_count = len(
        instance_ids
    )

    instance_areas = []

    for instance_id in instance_ids:

        area = int(
            np.sum(
                instance_mask == instance_id
            )
        )

        instance_areas.append(
            area
        )

    if len(instance_areas) > 0:

        min_instance_area = min(
            instance_areas
        )

        max_instance_area = max(
            instance_areas
        )

        mean_instance_area = np.mean(
            instance_areas
        )

        median_instance_area = np.median(
            instance_areas
        )

    else:

        min_instance_area = 0
        max_instance_area = 0
        mean_instance_area = 0
        median_instance_area = 0

    statistics = {
        "height": int(height),
        "width": int(width),
        "total_pixels": int(total_pixels),
        "tree_pixels": int(tree_pixels),
        "background_pixels": int(background_pixels),
        "tree_percentage": float(
            tree_percentage
        ),
        "background_percentage": float(
            background_percentage
        ),
        "instance_count": int(
            instance_count
        ),
        "min_instance_area_pixel": int(
            min_instance_area
        ),
        "max_instance_area_pixel": int(
            max_instance_area
        ),
        "mean_instance_area_pixel": float(
            mean_instance_area
        ),
        "median_instance_area_pixel": float(
            median_instance_area
        ),
        "tree_type": (
            "Tree - jenis belum teridentifikasi"
        ),
    }

    return statistics


# ============================================================
# MENYIMPAN CSV DETAIL SETIAP INSTANCE
# ============================================================

def save_instance_details_csv(
    instance_details,
    output_path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "instance_id",
        "tree_type",
        "area_pixels",
        "bbox_x",
        "bbox_y",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "centroid_x",
        "centroid_y",
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

        for detail in instance_details:

            writer.writerow(
                detail
            )

    print(
        f"CSV detail instance: {output_path}"
    )


# ============================================================
# MENYIMPAN RINGKASAN TXT
# ============================================================

def save_detail_report(
    image_path,
    statistics,
    instance_details,
    semantic_path,
    instance_path,
    semantic_overlay_path,
    bbox_overlay_path,
    semantic_preview_path,
    instance_preview_path,
    instance_csv_path,
    report_path,
):

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 75 + "\n")

        file.write(
            "LAPORAN DETAIL HASIL INFERENCE SEGFORMER-B5\n"
        )

        file.write("=" * 75 + "\n\n")

        file.write("INFORMASI CITRA\n")
        file.write("-" * 75 + "\n")

        file.write(
            f"Nama file       : {image_path.name}\n"
        )

        file.write(
            f"Lokasi input    : {image_path}\n"
        )

        file.write(
            f"Ukuran citra    : "
            f"{statistics['width']} x "
            f"{statistics['height']} piksel\n"
        )

        file.write(
            f"Total piksel    : "
            f"{statistics['total_pixels']:,}\n"
        )

        file.write(
            "\nHASIL SEMANTIC SEGMENTATION\n"
        )

        file.write("-" * 75 + "\n")

        file.write(
            f"Piksel pohon    : "
            f"{statistics['tree_pixels']:,}\n"
        )

        file.write(
            f"Piksel background: "
            f"{statistics['background_pixels']:,}\n"
        )

        file.write(
            f"Persentase pohon: "
            f"{statistics['tree_percentage']:.4f}%\n"
        )

        file.write(
            f"Persentase background: "
            f"{statistics['background_percentage']:.4f}%\n"
        )

        file.write(
            "\nHASIL INSTANCE SEGMENTATION\n"
        )

        file.write("-" * 75 + "\n")

        file.write(
            f"Jumlah instance : "
            f"{statistics['instance_count']:,}\n"
        )

        file.write(
            f"Luas instance minimum: "
            f"{statistics['min_instance_area_pixel']:,} piksel\n"
        )

        file.write(
            f"Luas instance maksimum: "
            f"{statistics['max_instance_area_pixel']:,} piksel\n"
        )

        file.write(
            f"Rata-rata luas instance: "
            f"{statistics['mean_instance_area_pixel']:.2f} piksel\n"
        )

        file.write(
            f"Median luas instance: "
            f"{statistics['median_instance_area_pixel']:.2f} piksel\n"
        )

        file.write(
            f"Jenis objek: "
            f"{statistics['tree_type']}\n"
        )

        file.write(
            "\nDETAIL SETIAP INSTANCE\n"
        )

        file.write("-" * 75 + "\n")

        if len(instance_details) == 0:

            file.write(
                "Tidak ada instance pohon yang terdeteksi.\n"
            )

        else:

            for detail in instance_details:

                file.write(
                    f"ID Instance       : "
                    f"{detail['instance_id']}\n"
                )

                file.write(
                    f"Jenis objek       : "
                    f"{detail['tree_type']}\n"
                )

                file.write(
                    f"Luas              : "
                    f"{detail['area_pixels']:,} piksel\n"
                )

                file.write(
                    f"Bounding box      : "
                    f"({detail['bbox_x']}, "
                    f"{detail['bbox_y']}) - "
                    f"({detail['bbox_x2']}, "
                    f"{detail['bbox_y2']})\n"
                )

                file.write(
                    f"Ukuran bbox       : "
                    f"{detail['bbox_width']} x "
                    f"{detail['bbox_height']} piksel\n"
                )

                file.write(
                    f"Centroid          : "
                    f"({detail['centroid_x']}, "
                    f"{detail['centroid_y']})\n"
                )

                file.write("-" * 75 + "\n")

        file.write("\nFILE OUTPUT\n")
        file.write("-" * 75 + "\n")

        file.write(
            f"Semantic mask       : {semantic_path}\n"
        )

        file.write(
            f"Instance mask       : {instance_path}\n"
        )

        file.write(
            f"Overlay semantic    : {semantic_overlay_path}\n"
        )

        file.write(
            f"Overlay bounding box: {bbox_overlay_path}\n"
        )

        file.write(
            f"Semantic preview    : "
            f"{semantic_preview_path}\n"
        )

        file.write(
            f"Instance preview    : "
            f"{instance_preview_path}\n"
        )

        file.write(
            f"CSV detail instance : "
            f"{instance_csv_path}\n"
        )

        file.write(
            "\n" + "=" * 75 + "\n"
        )

        file.write(
            "SELESAI\n"
        )

        file.write(
            "=" * 75 + "\n"
        )


# ============================================================
# MENYIMPAN STATISTIK CSV
# ============================================================

def save_statistics_csv(
    rows,
    output_path,
):

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


# ============================================================
# MENYIMPAN STATISTIK JSON
# ============================================================

def save_statistics_json(
    statistics,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# MAIN BATCH INFERENCE
# ============================================================

def main():

    create_all_directories()

    if not INPUT_TIF_DIR.exists():

        raise FileNotFoundError(
            f"Folder input tidak ditemukan: "
            f"{INPUT_TIF_DIR}"
        )

    image_paths = sorted(
        list(
            INPUT_TIF_DIR.glob("*.tif")
        )
        + list(
            INPUT_TIF_DIR.glob("*.tiff")
        )
    )

    if len(image_paths) == 0:

        raise FileNotFoundError(
            f"Tidak ditemukan file .tif atau .tiff "
            f"di folder: {INPUT_TIF_DIR}"
        )

    processor, model = load_model()

    print("=" * 75)
    print("BATCH INFERENCE SEGFORMER-B5")
    print("=" * 75)

    print(
        f"Folder input : {INPUT_TIF_DIR}"
    )

    print(
        f"Jumlah citra : {len(image_paths)}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print("=" * 75)

    all_statistics = []

    berhasil = 0
    gagal = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print("-" * 75)

        print(
            f"[{index}/{len(image_paths)}] "
            f"Memproses {image_path.name}"
        )

        print("-" * 75)

        try:

            # ------------------------------------------------
            # Baca citra
            # ------------------------------------------------

            image = read_tif(
                image_path
            )

            print(
                f"Ukuran citra: "
                f"{image.shape[1]} x "
                f"{image.shape[0]}"
            )

            # ------------------------------------------------
            # Prediksi semantic segmentation
            # ------------------------------------------------

            semantic_mask = predict_sliding_window(
                image,
                processor,
                model,
                patch_size=IMAGE_SIZE,
                stride=384,
            )

            # ------------------------------------------------
            # Konversi semantic ke instance
            # ------------------------------------------------

            instance_mask = semantic_to_instance(
                semantic_mask
            )

            # ------------------------------------------------
            # Ambil detail setiap instance
            # ------------------------------------------------

            instance_details = (
                extract_instance_details(
                    instance_mask
                )
            )

            # ------------------------------------------------
            # Buat statistik
            # ------------------------------------------------

            statistics = calculate_statistics(
                image,
                semantic_mask,
                instance_mask,
            )

            # ------------------------------------------------
            # Buat dua overlay
            # ------------------------------------------------

            # Overlay semantic tetap dipertahankan
            semantic_overlay = create_semantic_overlay(
                image,
                semantic_mask,
                statistics,
            )

            # Overlay utama:
            # citra asli + hijau transparan +
            # boundary merah + bbox merah + centroid
            bbox_overlay = create_bbox_overlay(
                image,
                semantic_mask,
                instance_mask,
            )

            # ------------------------------------------------
            # Buat preview
            # ------------------------------------------------

            semantic_preview = (
                create_semantic_preview(
                    semantic_mask
                )
            )

            instance_preview = (
                create_instance_preview(
                    instance_mask
                )
            )

            # ------------------------------------------------
            # Path output
            # ------------------------------------------------

            stem = image_path.stem

            semantic_path = (
                SEMANTIC_OUTPUT_DIR
                / f"{stem}_semantic.png"
            )

            instance_path = (
                INSTANCE_OUTPUT_DIR
                / f"{stem}_instance.tif"
            )

            semantic_overlay_path = (
                SEMANTIC_OVERLAY_DIR
                / f"{stem}_semantic_only.png"
            )

            bbox_overlay_path = (
                BBOX_OVERLAY_DIR
                / f"{stem}_bounding_box.png"
            )

            semantic_preview_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_semantic_preview.png"
            )

            instance_preview_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_instance_preview.png"
            )

            instance_csv_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_instances.csv"
            )

            report_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_report.txt"
            )

            statistics_json_path = (
                STATISTICS_OUTPUT_DIR
                / f"{stem}_statistics.json"
            )

            # ------------------------------------------------
            # Simpan semantic mask
            # ------------------------------------------------

            cv2.imwrite(
                str(semantic_path),
                (
                    semantic_mask * 255
                ).astype(np.uint8),
            )

            # ------------------------------------------------
            # Simpan instance mask
            # ------------------------------------------------

            cv2.imwrite(
                str(instance_path),
                instance_mask.astype(
                    np.int32
                ),
            )

            # ------------------------------------------------
            # Simpan overlay semantic
            # ------------------------------------------------

            cv2.imwrite(
                str(semantic_overlay_path),
                cv2.cvtColor(
                    semantic_overlay,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            # ------------------------------------------------
            # Simpan overlay utama
            # ------------------------------------------------

            cv2.imwrite(
                str(bbox_overlay_path),
                cv2.cvtColor(
                    bbox_overlay,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            # ------------------------------------------------
            # Simpan semantic preview
            # ------------------------------------------------

            cv2.imwrite(
                str(semantic_preview_path),
                cv2.cvtColor(
                    semantic_preview,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            # ------------------------------------------------
            # Simpan instance preview
            # ------------------------------------------------

            cv2.imwrite(
                str(instance_preview_path),
                cv2.cvtColor(
                    instance_preview,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            # ------------------------------------------------
            # Simpan CSV detail instance
            # ------------------------------------------------

            save_instance_details_csv(
                instance_details,
                instance_csv_path,
            )

            # ------------------------------------------------
            # Simpan laporan TXT
            # ------------------------------------------------

            save_detail_report(
                image_path,
                statistics,
                instance_details,
                semantic_path,
                instance_path,
                semantic_overlay_path,
                bbox_overlay_path,
                semantic_preview_path,
                instance_preview_path,
                instance_csv_path,
                report_path,
            )

            # ------------------------------------------------
            # Simpan statistik JSON
            # ------------------------------------------------

            save_statistics_json(
                statistics,
                statistics_json_path,
            )

            # ------------------------------------------------
            # Rekap statistik
            # ------------------------------------------------

            row = {
                "file": image_path.name,
                "width": statistics["width"],
                "height": statistics["height"],
                "total_pixels": statistics["total_pixels"],
                "tree_pixels": statistics["tree_pixels"],
                "background_pixels": statistics[
                    "background_pixels"
                ],
                "tree_percentage": round(
                    statistics[
                        "tree_percentage"
                    ],
                    4,
                ),
                "background_percentage": round(
                    statistics[
                        "background_percentage"
                    ],
                    4,
                ),
                "instance_count": statistics[
                    "instance_count"
                ],
                "min_instance_area_pixel": statistics[
                    "min_instance_area_pixel"
                ],
                "max_instance_area_pixel": statistics[
                    "max_instance_area_pixel"
                ],
                "mean_instance_area_pixel": round(
                    statistics[
                        "mean_instance_area_pixel"
                    ],
                    2,
                ),
                "median_instance_area_pixel": round(
                    statistics[
                        "median_instance_area_pixel"
                    ],
                    2,
                ),
                "tree_type": statistics[
                    "tree_type"
                ],
            }

            all_statistics.append(
                row
            )

            # ------------------------------------------------
            # Tampilkan hasil
            # ------------------------------------------------

            print("\nHASIL PREDIKSI")

            print(
                f"Persentase pohon : "
                f"{statistics['tree_percentage']:.2f}%"
            )

            print(
                f"Persentase background : "
                f"{statistics['background_percentage']:.2f}%"
            )

            print(
                f"Jumlah instance/pohon : "
                f"{statistics['instance_count']}"
            )

            print(
                f"Rata-rata luas instance : "
                f"{statistics['mean_instance_area_pixel']:.2f} piksel"
            )

            print(
                f"Jenis objek : "
                f"{statistics['tree_type']}"
            )

            print("\nOutput:")

            print(
                f"  Semantic mask       : "
                f"{semantic_path}"
            )

            print(
                f"  Instance mask       : "
                f"{instance_path}"
            )

            print(
                f"  Overlay semantic    : "
                f"{semantic_overlay_path}"
            )

            print(
                f"  Overlay bounding box: "
                f"{bbox_overlay_path}"
            )

            print(
                f"  Semantic preview    : "
                f"{semantic_preview_path}"
            )

            print(
                f"  Instance preview    : "
                f"{instance_preview_path}"
            )

            print(
                f"  CSV detail          : "
                f"{instance_csv_path}"
            )

            print(
                f"  Report              : "
                f"{report_path}"
            )

            print(
                f"  Statistik JSON      : "
                f"{statistics_json_path}"
            )

            berhasil += 1

            # ------------------------------------------------
            # Bersihkan memori
            # ------------------------------------------------

            del image
            del semantic_mask
            del instance_mask
            del instance_details
            del semantic_overlay
            del bbox_overlay
            del semantic_preview
            del instance_preview

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception as e:

            gagal += 1

            print(
                f"GAGAL memproses "
                f"{image_path.name}"
            )

            print(
                f"Error: {e}"
            )

    # ========================================================
    # SIMPAN REKAP SELURUH CITRA
    # ========================================================

    summary_csv_path = (
        STATISTICS_OUTPUT_DIR
        / "summary_inference.csv"
    )

    save_statistics_csv(
        all_statistics,
        summary_csv_path,
    )

    print()
    print("=" * 75)
    print("BATCH INFERENCE SELESAI")
    print("=" * 75)

    print(
        f"Berhasil       : {berhasil}"
    )

    print(
        f"Gagal          : {gagal}"
    )

    print(
        f"Total citra    : {len(image_paths)}"
    )

    print(
        f"Rekap CSV      : {summary_csv_path}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()