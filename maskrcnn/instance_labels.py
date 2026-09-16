from __future__ import annotations

from pathlib import Path
from PIL import Image
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    BASE_DIR
    / "tcd_dataset"
    / "dataset"
    / "713.tif"
)

SEMANTIC_MASK_PATH = (
    BASE_DIR
    / "segformer"
    / "outputs_single_713"
    / "prediction"
    / "semantic_prediction_713.tif"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "dataset"
)

IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
MASK_OUTPUT_DIR = OUTPUT_DIR / "masks"
PREVIEW_OUTPUT_DIR = OUTPUT_DIR / "previews"

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASK_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETER INSTANCE
# ============================================================

MIN_TREE_SIZE = 50

# Semakin kecil, semakin mudah pohon berdempetan dipisahkan.
MIN_TREE_DISTANCE = 8

# Semakin kecil, semakin banyak titik pusat pohon yang terdeteksi.
MIN_PEAK_HEIGHT = 2


# ============================================================
# NORMALISASI CITRA
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


# ============================================================
# MEMBACA CITRA
# ============================================================

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
# MEMBACA SEMANTIC MASK
# ============================================================

def read_semantic_mask(
    mask_path: Path,
) -> np.ndarray:

    with rasterio.open(mask_path) as src:
        semantic_mask = src.read(1)

    return semantic_mask.astype(
        np.uint8
    )


# ============================================================
# MEMBUAT INSTANCE MASK
# ============================================================

def create_instance_mask(
    semantic_mask: np.ndarray,
) -> np.ndarray:
    """
    Input:
        0 = Background
        1 = Tree

    Output:
        0 = Background
        1 = Tree 1
        2 = Tree 2
        3 = Tree 3
        dan seterusnya.
    """

    tree_mask = semantic_mask == 1

    tree_mask = remove_small_objects(
        tree_mask,
        min_size=MIN_TREE_SIZE,
        connectivity=2,
    )

    tree_mask = ndi.binary_fill_holes(
        tree_mask
    )

    if not np.any(tree_mask):
        return np.zeros_like(
            semantic_mask,
            dtype=np.int32,
        )

    distance = distance_transform_edt(
        tree_mask
    )

    coordinates = peak_local_max(
        distance,
        min_distance=MIN_TREE_DISTANCE,
        threshold_abs=MIN_PEAK_HEIGHT,
        labels=tree_mask,
        exclude_border=False,
    )

    markers = np.zeros_like(
        distance,
        dtype=np.int32,
    )

    for marker_id, coordinate in enumerate(
        coordinates,
        start=1,
    ):
        row, col = coordinate
        markers[row, col] = marker_id

    # Jika tidak ditemukan titik pusat,
    # gunakan connected component sebagai fallback.
    if len(coordinates) == 0:
        fallback_mask, _ = ndi.label(
            tree_mask,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        return fallback_mask.astype(
            np.int32
        )

    instance_mask = watershed(
        -distance,
        markers,
        mask=tree_mask,
        connectivity=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    # Bersihkan objek kecil dan susun ulang ID.
    final_mask = np.zeros_like(
        instance_mask,
        dtype=np.int32,
    )

    new_id = 1

    for instance_id in range(
        1,
        int(instance_mask.max()) + 1,
    ):
        object_mask = (
            instance_mask == instance_id
        )

        object_size = int(
            np.sum(object_mask)
        )

        if object_size < MIN_TREE_SIZE:
            continue

        final_mask[
            object_mask
        ] = new_id

        new_id += 1

    return final_mask


# ============================================================
# MENYIMPAN INSTANCE MASK
# ============================================================

def save_instance_mask(
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
# MEMBUAT BOUNDING BOX
# ============================================================

def create_boxes(
    instance_mask: np.ndarray,
) -> list[dict]:

    boxes = []

    number_of_instances = int(
        instance_mask.max()
    )

    for instance_id in range(
        1,
        number_of_instances + 1,
    ):
        rows, cols = np.where(
            instance_mask == instance_id
        )

        if len(rows) == 0:
            continue

        xmin = int(cols.min())
        ymin = int(rows.min())
        xmax = int(cols.max())
        ymax = int(rows.max())

        boxes.append(
            {
                "id": instance_id,
                "label": f"Tree {instance_id:03d}",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": xmax - xmin + 1,
                "height": ymax - ymin + 1,
                "area_pixels": int(len(rows)),
            }
        )

    return boxes


# ============================================================
# MENYIMPAN CSV
# ============================================================

def save_boxes_csv(
    boxes: list[dict],
    output_path: Path,
):

    fieldnames = [
        "id",
        "label",
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
        writer.writerows(boxes)


# ============================================================
# PREVIEW INSTANCE MASK DAN BOX
# ============================================================

def save_preview(
    image: np.ndarray,
    instance_mask: np.ndarray,
    boxes: list[dict],
    output_path: Path,
):

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

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8),
    )

    axes[0].imshow(image)
    axes[0].set_title(
        "Citra Asli"
    )
    axes[0].axis("off")

    axes[1].imshow(instance_rgb)
    axes[1].set_title(
        f"Instance Mask ({number_of_instances} pohon)"
    )
    axes[1].axis("off")

    axes[2].imshow(image)

    for box in boxes:
        rectangle = patches.Rectangle(
            (
                box["xmin"],
                box["ymin"],
            ),
            box["width"],
            box["height"],
            linewidth=1.2,
            edgecolor="red",
            facecolor="none",
        )

        axes[2].add_patch(
            rectangle
        )

        axes[2].text(
            box["xmin"],
            max(box["ymin"] - 3, 0),
            box["label"],
            color="yellow",
            fontsize=7,
            bbox={
                "facecolor": "black",
                "alpha": 0.7,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    axes[2].set_title(
        f"Bounding Box ({len(boxes)} pohon)"
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
    print("MEMBUAT INSTANCE LABEL UNTUK MASK R-CNN")
    print("=" * 80)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Citra tidak ditemukan: {IMAGE_PATH}"
        )

    if not SEMANTIC_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Semantic mask tidak ditemukan: {SEMANTIC_MASK_PATH}"
        )

    print("[1/5] Membaca citra...")
    image, profile = read_image(
        IMAGE_PATH
    )

    print(
        f"Ukuran citra: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print("[2/5] Membaca semantic mask...")
    semantic_mask = read_semantic_mask(
        SEMANTIC_MASK_PATH
    )

    print("[3/5] Membuat instance mask dengan watershed...")
    instance_mask = create_instance_mask(
        semantic_mask
    )

    number_of_instances = int(
        instance_mask.max()
    )

    print(
        f"Jumlah individu pohon: "
        f"{number_of_instances}"
    )

    image_output_path = (
        IMAGE_OUTPUT_DIR
        / "713.png"
    )

    mask_output_path = (
        MASK_OUTPUT_DIR
        / "713.tif"
    )

    preview_output_path = (
        PREVIEW_OUTPUT_DIR
        / "713_preview.png"
    )

    csv_output_path = (
        OUTPUT_DIR
        / "713_boxes.csv"
    )

    print("[4/5] Menyimpan citra dan instance mask...")

    Image.fromarray(
        image
    ).save(
        image_output_path
    )

    save_instance_mask(
        instance_mask,
        profile,
        mask_output_path,
    )

    boxes = create_boxes(
        instance_mask
    )

    save_boxes_csv(
        boxes,
        csv_output_path,
    )

    print("[5/5] Membuat preview...")
    save_preview(
        image,
        instance_mask,
        boxes,
        preview_output_path,
    )

    print()
    print("=" * 80)
    print("INSTANCE LABEL BERHASIL DIBUAT")
    print("=" * 80)
    print(f"Citra       : {image_output_path}")
    print(f"Mask        : {mask_output_path}")
    print(f"Preview     : {preview_output_path}")
    print(f"Bounding CSV: {csv_output_path}")
    print(f"Jumlah Tree : {len(boxes)}")
    print("=" * 80)


if __name__ == "__main__":
    main()