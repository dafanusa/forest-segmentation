from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import rasterio
import torch
import torchvision

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.transforms import functional as TF

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
)


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR.parent
    / "tcd_dataset"
    / "dataset"
    / "713.tif"
)

MODEL_PATH = (
    BASE_DIR
    / "outputs_single_713"
    / "best_model.pth"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs_single_713"
    / "prediction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

SCORE_THRESHOLD = 0.50
MASK_THRESHOLD = 0.50


# ============================================================
# MEMBACA CITRA
# ============================================================

def normalize_to_uint8(
    image: np.ndarray,
) -> np.ndarray:

    image = image.astype(np.float32)

    result = np.zeros_like(
        image,
        dtype=np.uint8,
    )

    for channel in range(image.shape[2]):

        band = image[:, :, channel]

        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        values = band[valid]

        low = np.percentile(
            values,
            2,
        )

        high = np.percentile(
            values,
            98,
        )

        if high <= low:
            low = values.min()
            high = values.max()

        if high <= low:
            continue

        scaled = (
            (band - low)
            / (high - low)
            * 255.0
        )

        scaled = np.clip(
            scaled,
            0,
            255,
        )

        scaled[~valid] = 0

        result[:, :, channel] = scaled.astype(
            np.uint8
        )

    return result


def read_image(
    image_path: Path,
):

    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile.copy()

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

    if image.shape[2] > 3:
        image = image[:, :, :3]

    image = normalize_to_uint8(
        image
    )

    return image, profile


# ============================================================
# MEMBUAT MODEL
# ============================================================

def create_model():

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features_box = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features_box,
            NUM_CLASSES,
        )
    )

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
# MEMUAT MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# PREDIKSI
# ============================================================

@torch.no_grad()
def predict(
    image: np.ndarray,
    model,
):

    image_tensor = TF.to_tensor(
        image
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    outputs = model(
        [
            image_tensor
        ]
    )

    output = outputs[0]

    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()

    keep = scores >= SCORE_THRESHOLD

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]

    # Mask R-CNN menghasilkan mask:
    # N x 1 x H x W
    masks = masks[:, 0]

    binary_masks = (
        masks >= MASK_THRESHOLD
    )

    return (
        boxes,
        labels,
        scores,
        binary_masks,
    )


# ============================================================
# MEMBUAT INSTANCE MASK DARI OUTPUT MASK R-CNN
# ============================================================

def create_instance_mask(
    binary_masks: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:

    height, width = image_shape

    instance_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.int32,
    )

    for instance_id, mask in enumerate(
        binary_masks,
        start=1,
    ):

        instance_mask[mask] = instance_id

    return instance_mask


# ============================================================
# MENYIMPAN INSTANCE GEOTIFF
# ============================================================

def save_instance_raster(
    instance_mask: np.ndarray,
    profile: dict,
    output_path: Path,
):

    output_profile = profile.copy()

    output_profile.pop(
        "nodata",
        None,
    )

    output_profile.update(
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
        **output_profile,
    ) as dst:

        dst.write(
            instance_mask.astype(
                np.int32
            ),
            1,
        )


# ============================================================
# MENYIMPAN CSV
# ============================================================

def save_boxes_csv(
    boxes: np.ndarray,
    scores: np.ndarray,
    binary_masks: np.ndarray,
    output_path: Path,
):

    fieldnames = [
        "id",
        "label",
        "score",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width",
        "height",
        "area_pixels",
    ]

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

        for index, (
            box,
            score,
            mask,
        ) in enumerate(
            zip(
                boxes,
                scores,
                binary_masks,
            ),
            start=1,
        ):

            xmin, ymin, xmax, ymax = box

            xmin = int(round(xmin))
            ymin = int(round(ymin))
            xmax = int(round(xmax))
            ymax = int(round(ymax))

            writer.writerow(
                {
                    "id": index,
                    "label": f"Tree {index:03d}",
                    "score": f"{float(score):.6f}",
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "width": xmax - xmin,
                    "height": ymax - ymin,
                    "area_pixels": int(np.sum(mask)),
                }
            )


# ============================================================
# VISUALISASI HASIL
# ============================================================

def save_prediction_preview(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    binary_masks: np.ndarray,
    output_path: Path,
):

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8),
    )

    # --------------------------------------------------------
    # Panel 1: citra asli
    # --------------------------------------------------------

    axes[0].imshow(image)
    axes[0].set_title(
        "Citra Asli"
    )
    axes[0].axis("off")

    # --------------------------------------------------------
    # Panel 2: instance mask
    # --------------------------------------------------------

    height, width = image.shape[:2]

    instance_mask = create_instance_mask(
        binary_masks,
        (
            height,
            width,
        ),
    )

    number_of_instances = int(
        instance_mask.max()
    )

    rng = np.random.default_rng(42)

    colors = rng.integers(
        30,
        240,
        size=(
            number_of_instances + 1,
            3,
        ),
        dtype=np.uint8,
    )

    colors[0] = [
        0,
        0,
        0,
    ]

    instance_rgb = colors[
        instance_mask
    ]

    axes[1].imshow(
        instance_rgb
    )

    axes[1].set_title(
        f"Instance Mask ({number_of_instances} pohon)"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Panel 3: bounding box
    # --------------------------------------------------------

    axes[2].imshow(image)

    for index, (
        box,
        score,
    ) in enumerate(
        zip(
            boxes,
            scores,
        ),
        start=1,
    ):

        xmin, ymin, xmax, ymax = box

        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        rectangle = patches.Rectangle(
            (
                xmin,
                ymin,
            ),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        axes[2].add_patch(
            rectangle
        )

        axes[2].text(
            xmin,
            max(ymin - 3, 0),
            f"Tree {index:03d} ({score:.2f})",
            color="yellow",
            fontsize=7,
            bbox={
                "facecolor": "black",
                "alpha": 0.75,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    axes[2].set_title(
        f"Mask R-CNN Bounding Box ({len(boxes)} pohon)"
    )

    axes[2].axis("off")

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PREDIKSI MASK R-CNN")
    print("=" * 80)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Citra  : {IMAGE_PATH}"
    )

    print(
        f"Model  : {MODEL_PATH}"
    )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Citra tidak ditemukan: {IMAGE_PATH}"
        )

    print("[1/5] Membaca citra...")
    image, profile = read_image(
        IMAGE_PATH
    )

    print(
        f"Ukuran citra: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print("[2/5] Memuat model...")
    model = load_model()

    print("Model berhasil dimuat.")

    print("[3/5] Menjalankan prediksi...")

    boxes, labels, scores, binary_masks = predict(
        image,
        model,
    )

    print(
        f"Jumlah objek terdeteksi: "
        f"{len(boxes)}"
    )

    instance_mask = create_instance_mask(
        binary_masks,
        image.shape[:2],
    )

    instance_mask_path = (
        OUTPUT_DIR
        / "maskrcnn_instance_mask_713.tif"
    )

    bounding_box_csv_path = (
        OUTPUT_DIR
        / "maskrcnn_bounding_boxes_713.csv"
    )

    preview_path = (
        OUTPUT_DIR
        / "maskrcnn_prediction_713.png"
    )

    print("[4/5] Menyimpan hasil...")

    save_instance_raster(
        instance_mask,
        profile,
        instance_mask_path,
    )

    save_boxes_csv(
        boxes,
        scores,
        binary_masks,
        bounding_box_csv_path,
    )

    print("[5/5] Membuat preview...")

    save_prediction_preview(
        image,
        boxes,
        scores,
        binary_masks,
        preview_path,
    )

    print()
    print("=" * 80)
    print("PREDIKSI MASK R-CNN BERHASIL")
    print("=" * 80)

    print(
        f"Instance mask : {instance_mask_path}"
    )

    print(
        f"Bounding CSV  : {bounding_box_csv_path}"
    )

    print(
        f"Preview       : {preview_path}"
    )

    print(
        f"Jumlah Tree   : {len(boxes)}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()