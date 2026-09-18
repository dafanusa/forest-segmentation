from __future__ import annotations

from pathlib import Path
import csv
import gc
import json
import math
import textwrap
import time

import cv2
import numpy as np
import rasterio

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
)


# ============================================================
# IMPORT CONFIG
# TIDAK MEMAKSA TREE_THRESHOLD HARUS ADA DI CONFIG
# ============================================================

try:
    from . import config_upernet as cfg
except ImportError:
    import upernet.config_upernet as cfg


MODEL_NAME = getattr(
    cfg,
    "MODEL_NAME",
    "openmmlab/upernet-swin-tiny",
)

NUM_CLASSES = getattr(
    cfg,
    "NUM_CLASSES",
    2,
)

ID2LABEL = getattr(
    cfg,
    "ID2LABEL",
    {
        0: "background",
        1: "tree",
    },
)

LABEL2ID = getattr(
    cfg,
    "LABEL2ID",
    {
        "background": 0,
        "tree": 1,
    },
)

BEST_MODEL_PATH = Path(
    getattr(
        cfg,
        "BEST_MODEL_PATH",
        Path("outputs/upernet/checkpoints/best_upernet.pth"),
    )
)

PREDICT_INPUT_DIR = Path(
    getattr(
        cfg,
        "PREDICT_INPUT_DIR",
        Path("tcd_dataset/dataset"),
    )
)

PREDICT_OUTPUT_DIR = Path(
    getattr(
        cfg,
        "PREDICT_OUTPUT_DIR",
        Path("outputs/upernet/predictions"),
    )
)

PREDICT_TILE_SIZE = int(
    getattr(
        cfg,
        "PREDICT_TILE_SIZE",
        512,
    )
)

PREDICT_OVERLAP = int(
    getattr(
        cfg,
        "PREDICT_OVERLAP",
        64,
    )
)

PREDICT_BATCH_SIZE = int(
    getattr(
        cfg,
        "PREDICT_BATCH_SIZE",
        1,
    )
)

TREE_THRESHOLD = float(
    getattr(
        cfg,
        "TREE_THRESHOLD",
        0.50,
    )
)

MIN_INSTANCE_AREA = int(
    getattr(
        cfg,
        "MIN_INSTANCE_AREA",
        100,
    )
)

WATERSHED_MIN_DISTANCE = int(
    getattr(
        cfg,
        "WATERSHED_MIN_DISTANCE",
        20,
    )
)

DEVICE = getattr(
    cfg,
    "DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


# ============================================================
# FOLDER OUTPUT
# ============================================================

SEMANTIC_OUTPUT_DIR = (
    PREDICT_OUTPUT_DIR / "semantic"
)

INSTANCE_OUTPUT_DIR = (
    PREDICT_OUTPUT_DIR / "instance"
)

SEMANTIC_OVERLAY_DIR = (
    PREDICT_OUTPUT_DIR / "overlay_semantic"
)

NORMAL_OVERLAY_DIR = (
    PREDICT_OUTPUT_DIR / "overlay_normal"
)

BBOX_OVERLAY_DIR = (
    PREDICT_OUTPUT_DIR / "overlay_bbox"
)

DETAIL_OUTPUT_DIR = (
    PREDICT_OUTPUT_DIR / "detail"
)

STATISTICS_OUTPUT_DIR = (
    PREDICT_OUTPUT_DIR / "statistics"
)

FULL_FRAME_OUTPUT_DIR = (
    PREDICT_OUTPUT_DIR / "full_frame"
)


# ============================================================
# MEMBUAT SEMUA DIREKTORI
# ============================================================

def create_all_directories():
    directories = [
        PREDICT_OUTPUT_DIR,
        SEMANTIC_OUTPUT_DIR,
        INSTANCE_OUTPUT_DIR,
        SEMANTIC_OVERLAY_DIR,
        NORMAL_OVERLAY_DIR,
        BBOX_OVERLAY_DIR,
        DETAIL_OUTPUT_DIR,
        STATISTICS_OUTPUT_DIR,
        FULL_FRAME_OUTPUT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# FONT UNTUK FULL FRAME
# ============================================================

def get_font(size=24, bold=False):
    font_candidates = []

    if bold:
        font_candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ]
    else:
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(
                    font_path,
                    size,
                )
            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# MEMBACA CITRA TIFF
# ============================================================

def read_tif(path: Path):
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

    elif image.shape[2] == 2:
        image = np.concatenate(
            [
                image,
                image[:, :, 1:2],
            ],
            axis=2,
        )

    elif image.shape[2] > 3:
        image = image[:, :, :3]

    image = image.astype(
        np.float32
    )

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
    else:
        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    return image.astype(
        np.uint8
    )


# ============================================================
# LOAD MODEL UPERNET
# ============================================================

def load_model():
    print("Memuat image processor UPerNet...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Memuat arsitektur UPerNet...")

    model = UperNetForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    print(
        f"Memuat checkpoint: {BEST_MODEL_PATH}"
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan:\n"
            f"{BEST_MODEL_PATH}"
        )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:
            state_dict = checkpoint[
                "model"
            ]

        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if len(missing_keys) > 0:
        print(
            f"Peringatan missing keys: "
            f"{len(missing_keys)}"
        )

    if len(unexpected_keys) > 0:
        print(
            f"Peringatan unexpected keys: "
            f"{len(unexpected_keys)}"
        )

    model.to(DEVICE)
    model.eval()

    print(
        f"Model UPerNet berhasil dimuat "
        f"pada device: {DEVICE}"
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

        logits = F.interpolate(
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

        tree_probability = probabilities[
            :, 1
        ]

        semantic_mask = (
            tree_probability
            > TREE_THRESHOLD
        )

        semantic_mask = (
            semantic_mask
            .cpu()
            .numpy()[0]
            .astype(np.uint8)
        )

    return semantic_mask


# ============================================================
# SLIDING WINDOW PREDICTION
# ============================================================

def predict_sliding_window(
    image,
    processor,
    model,
    patch_size=512,
    overlap=64,
):
    height, width = image.shape[:2]

    stride = patch_size - overlap

    if stride <= 0:
        raise ValueError(
            "Overlap harus lebih kecil "
            "daripada patch size."
        )

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

    y_positions = list(
        range(
            0,
            max(height - patch_size, 0) + 1,
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(width - patch_size, 0) + 1,
            stride,
        )
    )

    if len(y_positions) == 0:
        y_positions = [0]

    if len(x_positions) == 0:
        x_positions = [0]

    if y_positions[-1] + patch_size < height:
        y_positions.append(
            height - patch_size
        )

    if x_positions[-1] + patch_size < width:
        x_positions.append(
            width - patch_size
        )

    total_patches = (
        len(y_positions)
        * len(x_positions)
    )

    current_patch = 0

    for y in y_positions:
        for x in x_positions:
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

            patch_height, patch_width = (
                patch.shape[:2]
            )

            if (
                patch_height != patch_size
                or patch_width != patch_size
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
                    :patch_height,
                    :patch_width,
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
                :patch_height,
                :patch_width,
            ]

            prediction_sum[
                y:y2,
                x:x2,
            ] += semantic_patch

            prediction_count[
                y:y2,
                x:x2,
            ] += 1.0

            if (
                current_patch % 10 == 0
                or current_patch == total_patches
            ):
                print(
                    f"  Patch "
                    f"{current_patch}/"
                    f"{total_patches}",
                    end="\r",
                )

    print()

    semantic_mask = (
        prediction_sum
        / np.maximum(
            prediction_count,
            1.0,
        )
        > 0.5
    ).astype(np.uint8)

    return semantic_mask, total_patches


# ============================================================
# SEMANTIC KE INSTANCE MASK
# ============================================================

def semantic_to_instance(
    semantic_mask,
):
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
        binary_mask,
        dtype=np.uint8,
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

    for index, coordinate in enumerate(
        coordinates,
        start=1,
    ):
        row, col = coordinate

        markers[
            row,
            col,
        ] = index

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
# EKSTRAK DETAIL INSTANCE
# ============================================================

def extract_instance_details(
    instance_mask,
):
    details = []

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
            binary.sum()
        )

        if area_pixels <= 0:
            continue

        x, y, width, height = (
            cv2.boundingRect(binary)
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
                x + width / 2.0
            )

            centroid_y = (
                y + height / 2.0
            )

        details.append(
            {
                "instance_id": int(
                    instance_id
                ),
                "tree_type": (
                    "Tree - jenis "
                    "belum teridentifikasi"
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
        )

    return details


# ============================================================
# TEXT DENGAN BACKGROUND UNTUK CV2
# ============================================================

def draw_text_with_background(
    image,
    text,
    position,
    font_scale=0.45,
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
# OVERLAY SEMANTIC + PANEL STATISTIK
# ============================================================

def create_semantic_overlay(
    image,
    semantic_mask,
    statistics,
):
    overlay = image.copy()

    tree_area = (
        semantic_mask > 0
    )

    green_color = np.array(
        [
            0,
            255,
            0,
        ],
        dtype=np.float32,
    )

    alpha = 0.55

    overlay[
        tree_area
    ] = (
        (
            1.0 - alpha
        )
        * overlay[
            tree_area
        ].astype(np.float32)
        + alpha * green_color
    ).astype(np.uint8)

    panel_lines = [
        "UPERNET SEMANTIC SEGMENTATION",
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

    for index, line in enumerate(
        panel_lines
    ):
        draw_text_with_background(
            overlay,
            line,
            (
                panel_x,
                panel_y + index * 25,
            ),
            font_scale=(
                0.55
                if index == 0
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
            thickness=1,
            padding=4,
        )

    return overlay


# ============================================================
# OVERLAY NORMAL
# ============================================================

def create_normal_overlay(
    image,
    semantic_mask,
    alpha=0.35,
):
    overlay = image.copy()

    tree_area = (
        semantic_mask > 0
    )

    green_color = np.array(
        [
            0,
            255,
            0,
        ],
        dtype=np.float32,
    )

    original_pixels = (
        overlay[
            tree_area
        ].astype(np.float32)
    )

    blended_pixels = (
        (1.0 - alpha)
        * original_pixels
        + alpha
        * green_color
    )

    overlay[
        tree_area
    ] = blended_pixels.astype(
        np.uint8
    )

    return overlay


# ============================================================
# OVERLAY BBOX + BOUNDARY + CENTROID
# ============================================================

def create_bbox_overlay(
    image,
    semantic_mask,
    instance_mask,
):
    overlay = image.copy()

    tree_area = (
        semantic_mask > 0
    )

    green_color = np.array(
        [
            0,
            255,
            0,
        ],
        dtype=np.float32,
    )

    alpha = 0.35

    overlay[
        tree_area
    ] = (
        (
            1.0 - alpha
        )
        * overlay[
            tree_area
        ].astype(np.float32)
        + alpha * green_color
    ).astype(np.uint8)

    unique_instances = np.unique(
        instance_mask
    )

    for instance_id in unique_instances:
        if instance_id == 0:
            continue

        binary = (
            instance_mask == instance_id
        ).astype(np.uint8)

        x, y, width, height = (
            cv2.boundingRect(binary)
        )

        x2 = x + width - 1
        y2 = y + height - 1

        red_color = (
            255,
            0,
            0,
        )

        yellow_color = (
            255,
            255,
            0,
        )

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
                    x + width / 2.0
                )
            )

            centroid_y = int(
                round(
                    y + height / 2.0
                )
            )

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

        cv2.circle(
            overlay,
            (
                centroid_x,
                centroid_y,
            ),
            7,
            red_color,
            -1,
        )

        cv2.circle(
            overlay,
            (
                centroid_x,
                centroid_y,
            ),
            4,
            yellow_color,
            -1,
        )

        label = f"Tree {instance_id}"

        label_x = max(
            x,
            0,
        )

        label_y = max(
            y - 5,
            15,
        )

        draw_text_with_background(
            overlay,
            label,
            (
                label_x,
                label_y,
            ),
            font_scale=0.40,
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
            thickness=1,
            padding=3,
        )

    return overlay


# ============================================================
# PREVIEW SEMANTIC
# ============================================================

def create_semantic_preview(
    semantic_mask,
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
# PREVIEW INSTANCE
# ============================================================

def create_instance_preview(
    instance_mask,
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

    rng = np.random.default_rng(42)

    unique_instances = np.unique(
        instance_mask
    )

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
# HITUNG STATISTIK
# ============================================================

def calculate_statistics(
    image,
    semantic_mask,
    instance_mask,
):
    height, width = semantic_mask.shape

    total_pixels = (
        height * width
    )

    tree_pixels = int(
        np.sum(
            semantic_mask > 0
        )
    )

    background_pixels = (
        total_pixels - tree_pixels
    )

    tree_percentage = (
        tree_pixels
        / total_pixels
        * 100.0
        if total_pixels > 0
        else 0.0
    )

    background_percentage = (
        background_pixels
        / total_pixels
        * 100.0
        if total_pixels > 0
        else 0.0
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
                instance_mask
                == instance_id
            )
        )

        instance_areas.append(
            area
        )

    if len(instance_areas) > 0:
        min_area = min(
            instance_areas
        )

        max_area = max(
            instance_areas
        )

        mean_area = float(
            np.mean(
                instance_areas
            )
        )

        median_area = float(
            np.median(
                instance_areas
            )
        )
    else:
        min_area = 0
        max_area = 0
        mean_area = 0.0
        median_area = 0.0

    return {
        "height": int(height),
        "width": int(width),
        "total_pixels": int(
            total_pixels
        ),
        "tree_pixels": int(
            tree_pixels
        ),
        "background_pixels": int(
            background_pixels
        ),
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
            min_area
        ),
        "max_instance_area_pixel": int(
            max_area
        ),
        "mean_instance_area_pixel": float(
            mean_area
        ),
        "median_instance_area_pixel": float(
            median_area
        ),
        "tree_type": (
            "Tree - jenis "
            "belum teridentifikasi"
        ),
    }


# ============================================================
# SIMPAN INSTANCE MASK SEBAGAI TIFF
# ============================================================

def save_instance_tif(
    instance_mask,
    source_path,
    output_path,
):
    with rasterio.open(
        source_path
    ) as src:
        profile = src.profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "height": instance_mask.shape[0],
            "width": instance_mask.shape[1],
            "count": 1,
            "dtype": "int32",
            "compress": "lzw",
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            instance_mask.astype(
                np.int32
            ),
            1,
        )


# ============================================================
# SIMPAN PNG RGB
# ============================================================

def save_rgb_png(
    image,
    output_path,
):
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    Image.fromarray(
        image
    ).save(
        output_path
    )


# ============================================================
# SIMPAN CSV INSTANCE
# ============================================================

def save_instance_details_csv(
    details,
    output_path,
):
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
        writer.writerows(details)


# ============================================================
# SIMPAN JSON STATISTIK
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
# SIMPAN LAPORAN TXT
# ============================================================

def save_detail_report(
    image_path,
    statistics,
    details,
    output_paths,
    report_path,
):
    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("=" * 90 + "\n")
        file.write(
            "LAPORAN DETAIL INFERENCE UPERNET\n"
        )
        file.write("=" * 90 + "\n\n")

        file.write("INFORMASI CITRA\n")
        file.write("-" * 90 + "\n")
        file.write(
            f"Nama file        : {image_path.name}\n"
        )
        file.write(
            f"Lokasi input     : {image_path}\n"
        )
        file.write(
            f"Ukuran citra     : "
            f"{statistics['width']} x "
            f"{statistics['height']} piksel\n"
        )
        file.write(
            f"Total piksel     : "
            f"{statistics['total_pixels']:,}\n"
        )

        file.write(
            "\nHASIL SEMANTIC SEGMENTATION\n"
        )
        file.write("-" * 90 + "\n")
        file.write(
            f"Piksel pohon     : "
            f"{statistics['tree_pixels']:,}\n"
        )
        file.write(
            f"Piksel background: "
            f"{statistics['background_pixels']:,}\n"
        )
        file.write(
            f"Persentase pohon : "
            f"{statistics['tree_percentage']:.4f}%\n"
        )
        file.write(
            f"Persentase background: "
            f"{statistics['background_percentage']:.4f}%\n"
        )

        file.write(
            "\nHASIL INSTANCE SEGMENTATION\n"
        )
        file.write("-" * 90 + "\n")
        file.write(
            f"Jumlah instance  : "
            f"{statistics['instance_count']:,}\n"
        )
        file.write(
            f"Luas minimum     : "
            f"{statistics['min_instance_area_pixel']:,} piksel\n"
        )
        file.write(
            f"Luas maksimum    : "
            f"{statistics['max_instance_area_pixel']:,} piksel\n"
        )
        file.write(
            f"Rata-rata luas   : "
            f"{statistics['mean_instance_area_pixel']:.2f} piksel\n"
        )
        file.write(
            f"Median luas      : "
            f"{statistics['median_instance_area_pixel']:.2f} piksel\n"
        )
        file.write(
            f"Jenis objek      : "
            f"{statistics['tree_type']}\n"
        )

        file.write(
            "\nDETAIL INSTANCE\n"
        )
        file.write("-" * 90 + "\n")

        if len(details) == 0:
            file.write(
                "Tidak ada instance pohon yang terdeteksi.\n"
            )
        else:
            for detail in details:
                file.write(
                    f"Instance ID    : "
                    f"{detail['instance_id']}\n"
                )
                file.write(
                    f"Luas           : "
                    f"{detail['area_pixels']:,} piksel\n"
                )
                file.write(
                    f"Bounding box   : "
                    f"({detail['bbox_x']}, "
                    f"{detail['bbox_y']}) - "
                    f"({detail['bbox_x2']}, "
                    f"{detail['bbox_y2']})\n"
                )
                file.write(
                    f"Ukuran bbox    : "
                    f"{detail['bbox_width']} x "
                    f"{detail['bbox_height']} piksel\n"
                )
                file.write(
                    f"Centroid       : "
                    f"({detail['centroid_x']}, "
                    f"{detail['centroid_y']})\n"
                )
                file.write("-" * 90 + "\n")

        file.write("\nFILE OUTPUT\n")
        file.write("-" * 90 + "\n")

        for label, path in output_paths.items():
            file.write(
                f"{label:<30}: {path}\n"
            )


# ============================================================
# SIMPAN SUMMARY CSV
# ============================================================

def save_summary_csv(
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
# RESIZE PANEL TANPA MEMOTONG CITRA
# ============================================================

def resize_contain(
    image,
    target_width,
    target_height,
    background=(12, 12, 12),
):
    """
    Resize gambar agar masuk seluruhnya ke dalam panel.
    Tidak melakukan crop sehingga gambar tidak terpotong.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    original_height, original_width = (
        image.shape[:2]
    )

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(
        1,
        int(original_width * scale),
    )

    new_height = max(
        1,
        int(original_height * scale),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        background,
        dtype=np.uint8,
    )

    offset_x = (
        target_width - new_width
    ) // 2

    offset_y = (
        target_height - new_height
    ) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    return canvas


# ============================================================
# JUDUL PANEL FULL FRAME
# ============================================================

def add_panel_title(
    panel,
    title,
):
    panel = panel.copy()

    title_height = 48

    cv2.rectangle(
        panel,
        (
            0,
            0,
        ),
        (
            panel.shape[1],
            title_height,
        ),
        (
            20,
            20,
            20,
        ),
        -1,
    )

    cv2.putText(
        panel,
        title,
        (
            18,
            32,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return panel


# ============================================================
# PANEL STATISTIK FULL FRAME
# ============================================================

def create_statistics_panel(
    statistics,
    details,
    width,
    height,
):
    panel = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        (
            12,
            25,
            38,
        ),
    )

    draw = ImageDraw.Draw(
        panel
    )

    title_font = get_font(
        28,
        bold=True,
    )

    section_font = get_font(
        20,
        bold=True,
    )

    body_font = get_font(
        17,
        bold=False,
    )

    small_font = get_font(
        14,
        bold=False,
    )

    margin = 24

    draw.text(
        (
            margin,
            18,
        ),
        "STATISTIK HASIL PREDIKSI UPERNET",
        fill=(255, 255, 255),
        font=title_font,
    )

    draw.line(
        (
            margin,
            62,
            width - margin,
            62,
        ),
        fill=(100, 130, 150),
        width=2,
    )

    left_x = margin
    right_x = int(
        width * 0.52
    )

    y = 85

    statistics_lines = [
        (
            "Ukuran citra",
            f"{statistics['width']} x "
            f"{statistics['height']} px",
        ),
        (
            "Total piksel",
            f"{statistics['total_pixels']:,}",
        ),
        (
            "Piksel pohon",
            f"{statistics['tree_pixels']:,}",
        ),
        (
            "Piksel background",
            f"{statistics['background_pixels']:,}",
        ),
        (
            "Persentase pohon",
            f"{statistics['tree_percentage']:.2f}%",
        ),
        (
            "Persentase background",
            f"{statistics['background_percentage']:.2f}%",
        ),
        (
            "Jumlah instance",
            f"{statistics['instance_count']:,}",
        ),
        (
            "Luas instance minimum",
            f"{statistics['min_instance_area_pixel']:,} px",
        ),
        (
            "Luas instance maksimum",
            f"{statistics['max_instance_area_pixel']:,} px",
        ),
        (
            "Rata-rata luas instance",
            f"{statistics['mean_instance_area_pixel']:.2f} px",
        ),
        (
            "Median luas instance",
            f"{statistics['median_instance_area_pixel']:.2f} px",
        ),
        (
            "Jenis objek",
            statistics["tree_type"],
        ),
    ]

    for label, value in statistics_lines:
        draw.text(
            (
                left_x,
                y,
            ),
            f"{label:<27}:",
            fill=(230, 235, 240),
            font=body_font,
        )

        draw.text(
            (
                left_x + 300,
                y,
            ),
            str(value),
            fill=(255, 255, 255),
            font=body_font,
        )

        y += 32

    draw.text(
        (
            right_x,
            85,
        ),
        "DETAIL INSTANCE",
        fill=(255, 255, 255),
        font=section_font,
    )

    draw.line(
        (
            right_x,
            118,
            width - margin,
            118,
        ),
        fill=(100, 130, 150),
        width=1,
    )

    header_y = 135

    headers = [
        ("ID", right_x),
        ("Area (px)", right_x + 70),
        ("BBox", right_x + 220),
        ("Centroid", right_x + 520),
    ]

    for header, x in headers:
        draw.text(
            (
                x,
                header_y,
            ),
            header,
            fill=(220, 230, 240),
            font=small_font,
        )

    row_y = 165

    max_rows = max(
        1,
        int((height - 190) / 25),
    )

    visible_details = details[
        :max_rows
    ]

    for detail in visible_details:
        bbox_text = (
            f"({detail['bbox_x']},"
            f"{detail['bbox_y']})-"
            f"({detail['bbox_x2']},"
            f"{detail['bbox_y2']})"
        )

        centroid_text = (
            f"({detail['centroid_x']},"
            f"{detail['centroid_y']})"
        )

        values = [
            (
                str(detail["instance_id"]),
                right_x,
            ),
            (
                f"{detail['area_pixels']:,}",
                right_x + 70,
            ),
            (
                bbox_text,
                right_x + 220,
            ),
            (
                centroid_text,
                right_x + 520,
            ),
        ]

        for value, x in values:
            draw.text(
                (
                    x,
                    row_y,
                ),
                value,
                fill=(245, 245, 245),
                font=small_font,
            )

        row_y += 25

    if len(details) > max_rows:
        draw.text(
            (
                right_x,
                height - 34,
            ),
            f"... dan {len(details) - max_rows} instance lainnya",
            fill=(220, 220, 220),
            font=small_font,
        )

    return np.array(
        panel,
        dtype=np.uint8,
    )


# ============================================================
# MEMBUAT FULL FRAME GABUNGAN
# ============================================================

def create_full_frame(
    image,
    normal_overlay,
    semantic_preview,
    instance_preview,
    semantic_overlay,
    bbox_overlay,
    statistics,
    details,
    output_path,
):
    """
    Membuat satu frame gabungan seperti contoh user.

    Layout:
    Baris 1:
        Citra asli | Overlay normal | Semantic segmentation

    Baris 2:
        Instance segmentation | Overlay semantic + laporan |
        Overlay BBox + Boundary + Centroid

    Baris 3:
        Panel statistik dan detail instance
    """

    frame_width = 1800
    panel_width = 600
    panel_height = 430
    statistics_height = 430

    # Semua panel dibuat dengan rasio dan ukuran seragam.
    original_panel = resize_contain(
        image,
        panel_width,
        panel_height,
    )

    normal_panel = resize_contain(
        normal_overlay,
        panel_width,
        panel_height,
    )

    semantic_panel = resize_contain(
        semantic_preview,
        panel_width,
        panel_height,
    )

    instance_panel = resize_contain(
        instance_preview,
        panel_width,
        panel_height,
    )

    semantic_overlay_panel = resize_contain(
        semantic_overlay,
        panel_width,
        panel_height,
    )

    bbox_panel = resize_contain(
        bbox_overlay,
        panel_width,
        panel_height,
    )

    original_panel = add_panel_title(
        original_panel,
        "1. Citra Asli (TIF)",
    )

    normal_panel = add_panel_title(
        normal_panel,
        "2. Overlay Biasa",
    )

    semantic_panel = add_panel_title(
        semantic_panel,
        "3. Semantic Segmentation",
    )

    instance_panel = add_panel_title(
        instance_panel,
        "4. Instance Segmentation",
    )

    semantic_overlay_panel = add_panel_title(
        semantic_overlay_panel,
        "5. Overlay Semantic + Laporan",
    )

    bbox_panel = add_panel_title(
        bbox_panel,
        "6. Overlay BBox + Boundary + Centroid",
    )

    row_1 = np.concatenate(
        [
            original_panel,
            normal_panel,
            semantic_panel,
        ],
        axis=1,
    )

    row_2 = np.concatenate(
        [
            instance_panel,
            semantic_overlay_panel,
            bbox_panel,
        ],
        axis=1,
    )

    statistics_panel = create_statistics_panel(
        statistics,
        details,
        frame_width,
        statistics_height,
    )

    final_frame = np.concatenate(
        [
            row_1,
            row_2,
            statistics_panel,
        ],
        axis=0,
    )

    save_rgb_png(
        final_frame,
        output_path,
    )

    return final_frame


# ============================================================
# MAIN
# ============================================================

def main():
    create_all_directories()

    if not PREDICT_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Folder input tidak ditemukan:\n"
            f"{PREDICT_INPUT_DIR}"
        )

    image_paths = sorted(
        list(
            PREDICT_INPUT_DIR.glob("*.tif")
        )
        + list(
            PREDICT_INPUT_DIR.glob("*.tiff")
        )
    )

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Tidak ditemukan file TIFF di:\n"
            f"{PREDICT_INPUT_DIR}"
        )

    processor, model = load_model()

    print("=" * 90)
    print("BATCH DETAIL INFERENCE UPERNET")
    print("=" * 90)
    print(
        f"Folder input : {PREDICT_INPUT_DIR}"
    )
    print(
        f"Jumlah citra : {len(image_paths)}"
    )
    print(
        f"Checkpoint   : {BEST_MODEL_PATH}"
    )
    print(
        f"Device       : {DEVICE}"
    )
    print(
        f"Tile size    : {PREDICT_TILE_SIZE}"
    )
    print(
        f"Overlap      : {PREDICT_OVERLAP}"
    )
    print(
        f"Tree threshold: {TREE_THRESHOLD}"
    )
    print(
        f"Min instance area: {MIN_INSTANCE_AREA}"
    )
    print("=" * 90)

    all_statistics = []

    berhasil = 0
    gagal = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print()
        print("-" * 90)
        print(
            f"[{index}/{len(image_paths)}] "
            f"Memproses: {image_path.name}"
        )
        print("-" * 90)

        start_time = time.time()

        try:
            # ------------------------------------------------
            # BACA CITRA
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
            # PREDIKSI SEMANTIC
            # ------------------------------------------------

            semantic_mask, total_patches = (
                predict_sliding_window(
                    image,
                    processor,
                    model,
                    patch_size=PREDICT_TILE_SIZE,
                    overlap=PREDICT_OVERLAP,
                )
            )

            # ------------------------------------------------
            # KONVERSI INSTANCE
            # ------------------------------------------------

            instance_mask = (
                semantic_to_instance(
                    semantic_mask
                )
            )

            instance_details = (
                extract_instance_details(
                    instance_mask
                )
            )

            # ------------------------------------------------
            # HITUNG STATISTIK
            # ------------------------------------------------

            statistics = (
                calculate_statistics(
                    image,
                    semantic_mask,
                    instance_mask,
                )
            )

            statistics[
                "total_patches"
            ] = int(total_patches)

            statistics[
                "tree_threshold"
            ] = float(TREE_THRESHOLD)

            statistics[
                "min_instance_area"
            ] = int(MIN_INSTANCE_AREA)

            statistics[
                "watershed_min_distance"
            ] = int(WATERSHED_MIN_DISTANCE)

            # ------------------------------------------------
            # BUAT VISUALISASI
            # ------------------------------------------------

            semantic_overlay = (
                create_semantic_overlay(
                    image,
                    semantic_mask,
                    statistics,
                )
            )

            normal_overlay = (
                create_normal_overlay(
                    image,
                    semantic_mask,
                    alpha=0.35,
                )
            )

            bbox_overlay = (
                create_bbox_overlay(
                    image,
                    semantic_mask,
                    instance_mask,
                )
            )

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

            stem = image_path.stem

            # ------------------------------------------------
            # PATH OUTPUT
            # ------------------------------------------------

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
                / f"{stem}_semantic_overlay.png"
            )

            normal_overlay_path = (
                NORMAL_OVERLAY_DIR
                / f"{stem}_normal_overlay.png"
            )

            bbox_overlay_path = (
                BBOX_OVERLAY_DIR
                / f"{stem}_bbox_overlay.png"
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

            full_frame_path = (
                FULL_FRAME_OUTPUT_DIR
                / f"{stem}_full_frame.png"
            )

            # ------------------------------------------------
            # SIMPAN OUTPUT INDIVIDUAL
            # ------------------------------------------------

            cv2.imwrite(
                str(semantic_path),
                (
                    semantic_mask * 255
                ).astype(np.uint8),
            )

            save_instance_tif(
                instance_mask,
                image_path,
                instance_path,
            )

            save_rgb_png(
                semantic_overlay,
                semantic_overlay_path,
            )

            save_rgb_png(
                normal_overlay,
                normal_overlay_path,
            )

            save_rgb_png(
                bbox_overlay,
                bbox_overlay_path,
            )

            save_rgb_png(
                semantic_preview,
                semantic_preview_path,
            )

            save_rgb_png(
                instance_preview,
                instance_preview_path,
            )

            save_instance_details_csv(
                instance_details,
                instance_csv_path,
            )

            save_statistics_json(
                statistics,
                statistics_json_path,
            )

            output_paths = {
                "Semantic mask": semantic_path,
                "Instance mask": instance_path,
                "Semantic overlay": semantic_overlay_path,
                "Overlay biasa": normal_overlay_path,
                "Bounding box overlay": bbox_overlay_path,
                "Semantic preview": semantic_preview_path,
                "Instance preview": instance_preview_path,
                "CSV detail": instance_csv_path,
                "Statistics JSON": statistics_json_path,
                "Full frame": full_frame_path,
            }

            save_detail_report(
                image_path,
                statistics,
                instance_details,
                output_paths,
                report_path,
            )

            # ------------------------------------------------
            # BUAT SATU FRAME GABUNGAN
            # ------------------------------------------------

            create_full_frame(
                image=image,
                normal_overlay=normal_overlay,
                semantic_preview=semantic_preview,
                instance_preview=instance_preview,
                semantic_overlay=semantic_overlay,
                bbox_overlay=bbox_overlay,
                statistics=statistics,
                details=instance_details,
                output_path=full_frame_path,
            )

            elapsed_time = (
                time.time() - start_time
            )

            # ------------------------------------------------
            # REKAP STATISTIK
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
                "total_patches": total_patches,
                "tree_threshold": TREE_THRESHOLD,
                "processing_time_seconds": round(
                    elapsed_time,
                    2,
                ),
            }

            all_statistics.append(
                row
            )

            # ------------------------------------------------
            # PRINT HASIL
            # ------------------------------------------------

            print()
            print("HASIL PREDIKSI")
            print(
                f"Persentase pohon       : "
                f"{statistics['tree_percentage']:.2f}%"
            )
            print(
                f"Persentase background  : "
                f"{statistics['background_percentage']:.2f}%"
            )
            print(
                f"Jumlah instance/pohon  : "
                f"{statistics['instance_count']}"
            )
            print(
                f"Rata-rata luas instance: "
                f"{statistics['mean_instance_area_pixel']:.2f} piksel"
            )
            print(
                f"Jumlah patch           : "
                f"{total_patches}"
            )
            print(
                f"Waktu proses           : "
                f"{elapsed_time:.2f} detik"
            )

            print()
            print("OUTPUT:")

            for label, path in output_paths.items():
                print(
                    f"  {label:<30}: {path}"
                )

            print(
                f"  {'Report TXT':<30}: "
                f"{report_path}"
            )

            berhasil += 1

            # ------------------------------------------------
            # BERSIHKAN MEMORY
            # ------------------------------------------------

            del image
            del semantic_mask
            del instance_mask
            del instance_details
            del semantic_overlay
            del normal_overlay
            del bbox_overlay
            del semantic_preview
            del instance_preview

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as error:
            gagal += 1

            print()
            print(
                f"GAGAL memproses: "
                f"{image_path.name}"
            )

            print(
                f"Error: {error}"
            )

    # ========================================================
    # SIMPAN SUMMARY
    # ========================================================

    summary_csv_path = (
        STATISTICS_OUTPUT_DIR
        / "summary_inference.csv"
    )

    save_summary_csv(
        all_statistics,
        summary_csv_path,
    )

    print()
    print("=" * 90)
    print("BATCH DETAIL INFERENCE UPERNET SELESAI")
    print("=" * 90)
    print(
        f"Berhasil    : {berhasil}"
    )
    print(
        f"Gagal       : {gagal}"
    )
    print(
        f"Total citra : {len(image_paths)}"
    )
    print(
        f"Rekap CSV   : {summary_csv_path}"
    )
    print(
        f"Full frame  : {FULL_FRAME_OUTPUT_DIR}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()