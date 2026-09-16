from __future__ import annotations

from pathlib import Path
import csv
import warnings

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from PIL import Image

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


warnings.filterwarnings("ignore")


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

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs_single_713"
)

MODEL_DIR = OUTPUT_DIR / "best_model"
PREDICTION_DIR = OUTPUT_DIR / "prediction"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PENGATURAN PEMISAHAN POHON
# ============================================================

# Piksel minimum agar objek dianggap sebagai pohon.
MIN_TREE_SIZE = 50

# Jarak minimum antar titik pusat pohon.
# Semakin kecil, semakin banyak pohon yang dapat dipisahkan.
# Semakin besar, objek kecil akan lebih mudah digabung.
MIN_TREE_DISTANCE = 8

# Tinggi minimum puncak distance transform.
# Nilai lebih tinggi membuat deteksi pusat pohon lebih selektif.
MIN_PEAK_HEIGHT = 2

# Ukuran struktur morfologi untuk membersihkan mask.
MORPHOLOGY_STRUCTURE_SIZE = 3

# Warna bounding box.
BOUNDING_BOX_COLOR = "red"

# Transparansi overlay.
SEMANTIC_OVERLAY_ALPHA = 0.45
INSTANCE_OVERLAY_ALPHA = 0.55


# ============================================================
# NORMALISASI CITRA
# ============================================================

def normalize_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Mengubah citra menjadi uint8 dengan rentang 0-255.

    Format input:
        H x W x C
    """

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
            result[:, :, channel] = 0
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
) -> tuple[np.ndarray, dict]:
    """
    Membaca citra raster menggunakan rasterio.

    Output:
        image   : H x W x C
        profile : metadata raster
    """

    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile.copy()

    # C x H x W menjadi H x W x C
    image = np.transpose(
        image,
        (1, 2, 0),
    )

    # Jika hanya satu channel,
    # ubah menjadi tiga channel.
    if image.shape[2] == 1:
        image = np.repeat(
            image,
            3,
            axis=2,
        )

    # Jika lebih dari tiga channel,
    # gunakan tiga channel pertama.
    if image.shape[2] > 3:
        image = image[:, :, :3]

    image_uint8 = normalize_to_uint8(
        image
    )

    return image_uint8, profile


# ============================================================
# MEMUAT MODEL SEGFORMER
# ============================================================

def load_model():
    """
    Memuat model SegFormer hasil training.
    """

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Folder model tidak ditemukan:\n"
            f"{MODEL_DIR}"
        )

    processor = (
        SegformerImageProcessor.from_pretrained(
            str(MODEL_DIR)
        )
    )

    model = (
        SegformerForSemanticSegmentation.from_pretrained(
            str(MODEL_DIR)
        )
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# PREDIKSI SEMANTIC SEGMENTATION
# ============================================================

@torch.no_grad()
def predict_segmentation(
    image: np.ndarray,
    processor,
    model,
) -> np.ndarray:
    """
    Menghasilkan semantic mask.

    Kelas:
        0 = Background
        1 = Tree
    """

    original_height, original_width = image.shape[:2]

    inputs = processor(
        images=Image.fromarray(image),
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    logits = outputs.logits

    # Mengembalikan ukuran logits ke ukuran citra asli.
    logits = F.interpolate(
        logits,
        size=(
            original_height,
            original_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    semantic_mask = (
        prediction[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return semantic_mask


# ============================================================
# MEMBERSIHKAN MASK TREE
# ============================================================

def clean_tree_mask(
    semantic_mask: np.ndarray,
) -> np.ndarray:
    """
    Membersihkan mask kelas Tree.

    Kelas:
        0 = Background
        1 = Tree
    """

    tree_mask = semantic_mask == 1

    # Menghilangkan objek kecil.
    tree_mask = remove_small_objects(
        tree_mask,
        min_size=MIN_TREE_SIZE,
        connectivity=2,
    )

    # Mengisi lubang kecil di dalam objek.
    tree_mask = ndi.binary_fill_holes(
        tree_mask
    )

    # Operasi morfologi ringan.
    structure = np.ones(
        (
            MORPHOLOGY_STRUCTURE_SIZE,
            MORPHOLOGY_STRUCTURE_SIZE,
        ),
        dtype=bool,
    )

    tree_mask = ndi.binary_opening(
        tree_mask,
        structure=structure,
    )

    tree_mask = ndi.binary_closing(
        tree_mask,
        structure=structure,
    )

    cleaned_mask = np.zeros_like(
        semantic_mask,
        dtype=np.uint8,
    )

    cleaned_mask[tree_mask] = 1

    return cleaned_mask


# ============================================================
# MEMBUAT INSTANCE MASK DENGAN WATERSHED
# ============================================================

def create_instance_mask(
    semantic_mask: np.ndarray,
) -> np.ndarray:
    """
    Memisahkan individu pohon menggunakan watershed.

    Output:
        0 = Background
        1 = Tree 1
        2 = Tree 2
        3 = Tree 3
        dan seterusnya.

    Catatan:
        Ini merupakan pendekatan instance approximation.
        Hasil sangat bergantung pada kualitas semantic mask.
    """

    tree_mask = semantic_mask == 1

    if not np.any(tree_mask):
        return np.zeros_like(
            semantic_mask,
            dtype=np.int32,
        )

    # --------------------------------------------------------
    # 1. Distance transform
    # --------------------------------------------------------
    # Setiap piksel Tree memiliki nilai jarak
    # terhadap background terdekat.
    distance = distance_transform_edt(
        tree_mask
    )

    # --------------------------------------------------------
    # 2. Mencari titik pusat/puncak setiap pohon
    # --------------------------------------------------------
    coordinates = peak_local_max(
        distance,
        min_distance=MIN_TREE_DISTANCE,
        threshold_abs=MIN_PEAK_HEIGHT,
        labels=tree_mask,
        exclude_border=False,
    )

    # --------------------------------------------------------
    # 3. Membuat marker
    # --------------------------------------------------------
    markers = np.zeros_like(
        distance,
        dtype=np.int32,
    )

    if len(coordinates) == 0:
        # Fallback jika tidak ditemukan titik puncak.
        labeled_fallback, _ = ndi.label(
            tree_mask,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        return labeled_fallback.astype(
            np.int32
        )

    for marker_id, coordinate in enumerate(
        coordinates,
        start=1,
    ):
        row, col = coordinate
        markers[row, col] = marker_id

    # --------------------------------------------------------
    # 4. Watershed
    # --------------------------------------------------------
    instance_mask = watershed(
        -distance,
        markers,
        mask=tree_mask,
        connectivity=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    # --------------------------------------------------------
    # 5. Membersihkan objek terlalu kecil
    # --------------------------------------------------------
    cleaned_instance_mask = np.zeros_like(
        instance_mask,
        dtype=np.int32,
    )

    new_instance_id = 1

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

        cleaned_instance_mask[
            object_mask
        ] = new_instance_id

        new_instance_id += 1

    return cleaned_instance_mask


# ============================================================
# MENYIMPAN RASTER GEOTIFF
# ============================================================

def save_raster(
    output_path: Path,
    data: np.ndarray,
    profile: dict,
    dtype: str,
):
    """
    Menyimpan hasil mask sebagai GeoTIFF.
    """

    output_profile = profile.copy()

    # Metadata yang tidak diperlukan untuk mask.
    output_profile.pop(
        "nodata",
        None,
    )

    output_profile.update(
        {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": dtype,
            "compress": "lzw",
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:
        dst.write(
            data.astype(dtype),
            1,
        )


# ============================================================
# VISUALISASI SEMANTIC SEGMENTATION
# ============================================================

def save_semantic_visualization(
    image: np.ndarray,
    semantic_mask: np.ndarray,
    output_path: Path,
):
    """
    Menyimpan visualisasi semantic segmentation.
    """

    tree_mask = semantic_mask == 1

    total_pixels = semantic_mask.size

    background_pixels = np.sum(
        semantic_mask == 0
    )

    tree_pixels = np.sum(
        semantic_mask == 1
    )

    background_percentage = (
        background_pixels
        / total_pixels
        * 100
    )

    tree_percentage = (
        tree_pixels
        / total_pixels
        * 100
    )

    semantic_cmap = ListedColormap(
        [
            "black",
            "white",
        ]
    )

    overlay = image.astype(
        np.float32
    ).copy()

    green_color = np.array(
        [0, 220, 80],
        dtype=np.float32,
    )

    overlay[tree_mask] = (
        (
            1
            - SEMANTIC_OVERLAY_ALPHA
        )
        * overlay[tree_mask]
        + SEMANTIC_OVERLAY_ALPHA
        * green_color
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7),
    )

    figure.suptitle(
        "SegFormer - Semantic Segmentation",
        fontsize=18,
        fontweight="bold",
    )

    axes[0].imshow(image)
    axes[0].set_title(
        "Citra Asli",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(
        semantic_mask,
        cmap=semantic_cmap,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        "Mask Tree dan Background",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Overlay SegFormer",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].axis("off")

    figure.text(
        0.5,
        0.02,
        (
            f"Background: "
            f"{background_percentage:.2f}%"
            "    |    "
            f"Tree: "
            f"{tree_percentage:.2f}%"
        ),
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.93,
        ]
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# VISUALISASI INSTANCE MASK
# ============================================================

def save_instance_visualization(
    image: np.ndarray,
    instance_mask: np.ndarray,
    output_path: Path,
):
    """
    Menyimpan visualisasi individu pohon
    dengan warna berbeda.
    """

    number_of_instances = int(
        instance_mask.max()
    )

    if number_of_instances == 0:
        instance_rgb = np.zeros_like(
            image,
            dtype=np.uint8,
        )
    else:
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

    tree_pixels = instance_mask > 0

    overlay = image.astype(
        np.float32
    ).copy()

    overlay[tree_pixels] = (
        (
            1
            - INSTANCE_OVERLAY_ALPHA
        )
        * overlay[tree_pixels]
        + INSTANCE_OVERLAY_ALPHA
        * instance_rgb[tree_pixels]
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7),
    )

    figure.suptitle(
        "SegFormer - Instance Approximation dengan Watershed",
        fontsize=18,
        fontweight="bold",
    )

    axes[0].imshow(image)
    axes[0].set_title(
        "Citra Asli",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(instance_rgb)
    axes[1].set_title(
        (
            "Pemisahan Individu Pohon "
            f"({number_of_instances} objek)"
        ),
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Overlay Instance",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].axis("off")

    figure.text(
        0.5,
        0.02,
        (
            "Setiap warna menunjukkan "
            "satu individu pohon"
        ),
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.93,
        ]
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MEMBUAT BOUNDING BOX PER POHON
# ============================================================

def create_tree_bounding_boxes(
    instance_mask: np.ndarray,
) -> list[dict]:
    """
    Membuat satu bounding box untuk setiap individu pohon.
    """

    number_of_instances = int(
        instance_mask.max()
    )

    boxes = []

    for instance_id in range(
        1,
        number_of_instances + 1,
    ):
        rows, cols = np.where(
            instance_mask == instance_id
        )

        if len(rows) == 0 or len(cols) == 0:
            continue

        xmin = int(cols.min())
        ymin = int(rows.min())
        xmax = int(cols.max())
        ymax = int(rows.max())

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        area_pixels = int(
            len(rows)
        )

        boxes.append(
            {
                "id": instance_id,
                "label": f"Tree {instance_id:03d}",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height,
                "area_pixels": area_pixels,
            }
        )

    return boxes


# ============================================================
# VISUALISASI BOUNDING BOX
# ============================================================

def save_tree_bounding_boxes(
    image: np.ndarray,
    instance_mask: np.ndarray,
    output_path: Path,
) -> list[dict]:
    """
    Membuat bounding box untuk setiap individu pohon.
    """

    boxes = create_tree_bounding_boxes(
        instance_mask
    )

    figure, ax = plt.subplots(
        figsize=(18, 12),
    )

    ax.imshow(image)

    for box in boxes:
        xmin = box["xmin"]
        ymin = box["ymin"]
        width = box["width"]
        height = box["height"]

        rectangle = patches.Rectangle(
            (
                xmin,
                ymin,
            ),
            width,
            height,
            linewidth=1.2,
            edgecolor=BOUNDING_BOX_COLOR,
            facecolor="none",
        )

        ax.add_patch(rectangle)

        # Label ditempatkan di bagian atas kotak.
        label_y = max(
            ymin - 3,
            0,
        )

        ax.text(
            xmin,
            label_y,
            box["label"],
            color="yellow",
            fontsize=7,
            fontweight="bold",
            bbox={
                "facecolor": "black",
                "alpha": 0.75,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    ax.set_title(
        "SegFormer - Bounding Box Setiap Individu Pohon",
        fontsize=18,
        fontweight="bold",
    )

    ax.axis("off")

    figure.text(
        0.5,
        0.02,
        (
            f"Jumlah individu pohon terdeteksi: "
            f"{len(boxes)}"
        ),
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.96,
        ]
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return boxes


# ============================================================
# MENYIMPAN KOORDINAT BOUNDING BOX KE CSV
# ============================================================

def save_boxes_to_csv(
    boxes: list[dict],
    output_path: Path,
):
    """
    Menyimpan koordinat bounding box ke CSV.
    """

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
# MAIN PROGRAM
# ============================================================

def main():
    print("=" * 80)
    print("SEGFORMER - PREDIKSI INDIVIDU POHON DAN BOUNDING BOX")
    print("=" * 80)

    print(f"Perangkat       : {DEVICE}")
    print(f"Citra input     : {IMAGE_PATH}")
    print(f"Model           : {MODEL_DIR}")
    print(f"Folder output   : {PREDICTION_DIR}")
    print()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "Citra input tidak ditemukan:\n"
            f"{IMAGE_PATH}"
        )

    print("[1/8] Membaca citra...")
    image, profile = read_image(
        IMAGE_PATH
    )

    height, width = image.shape[:2]

    print(
        f"Ukuran citra    : "
        f"{width} x {height}"
    )

    print(
        f"Jumlah channel  : "
        f"{image.shape[2]}"
    )
    print()

    print("[2/8] Memuat model SegFormer...")
    processor, model = load_model()
    print("Model berhasil dimuat.")
    print()

    print("[3/8] Menjalankan prediksi semantic segmentation...")
    semantic_mask = predict_segmentation(
        image,
        processor,
        model,
    )

    print(
        "Prediksi semantic segmentation selesai."
    )
    print()

    print("[4/8] Membersihkan mask Tree...")
    semantic_mask = clean_tree_mask(
        semantic_mask
    )

    tree_pixel_count = int(
        np.sum(semantic_mask == 1)
    )

    print(
        f"Jumlah piksel Tree: "
        f"{tree_pixel_count}"
    )
    print()

    print("[5/8] Memisahkan individu pohon dengan watershed...")
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
    print()

    semantic_tif_path = (
        PREDICTION_DIR
        / "semantic_prediction_713.tif"
    )

    instance_tif_path = (
        PREDICTION_DIR
        / "instance_prediction_713.tif"
    )

    semantic_png_path = (
        PREDICTION_DIR
        / "semantic_segmentation_713.png"
    )

    instance_png_path = (
        PREDICTION_DIR
        / "instance_segmentation_713.png"
    )

    bounding_box_png_path = (
        PREDICTION_DIR
        / "tree_bounding_boxes_713.png"
    )

    bounding_box_csv_path = (
        PREDICTION_DIR
        / "tree_bounding_boxes_713.csv"
    )

    print("[6/8] Menyimpan GeoTIFF...")

    save_raster(
        semantic_tif_path,
        semantic_mask,
        profile,
        dtype="uint8",
    )

    save_raster(
        instance_tif_path,
        instance_mask,
        profile,
        dtype="int32",
    )

    print("[7/8] Membuat visualisasi...")

    save_semantic_visualization(
        image,
        semantic_mask,
        semantic_png_path,
    )

    save_instance_visualization(
        image,
        instance_mask,
        instance_png_path,
    )

    tree_boxes = save_tree_bounding_boxes(
        image,
        instance_mask,
        bounding_box_png_path,
    )

    print("[8/8] Menyimpan koordinat bounding box...")

    save_boxes_to_csv(
        tree_boxes,
        bounding_box_csv_path,
    )

    print()
    print("=" * 80)
    print("PREDIKSI INDIVIDU POHON BERHASIL")
    print("=" * 80)

    print(
        f"Semantic GeoTIFF : "
        f"{semantic_tif_path}"
    )

    print(
        f"Instance GeoTIFF : "
        f"{instance_tif_path}"
    )

    print(
        f"Semantic PNG     : "
        f"{semantic_png_path}"
    )

    print(
        f"Instance PNG     : "
        f"{instance_png_path}"
    )

    print(
        f"Bounding Box PNG : "
        f"{bounding_box_png_path}"
    )

    print(
        f"Bounding Box CSV : "
        f"{bounding_box_csv_path}"
    )

    print(
        f"Jumlah Tree      : "
        f"{len(tree_boxes)}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()