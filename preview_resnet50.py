from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling


# ============================================================
# KONFIGURASI
# ============================================================

IMAGE_NAME = "Dan_2014_RGB_project_to_CHM.tif"

MASK_NAME = "Danum_resnet50_mask.tif"

PROBABILITY_NAME = "Danum_resnet50_probability.tif"

MAX_PREVIEW_PIXELS = 2400

# Transparansi area segmentasi
MASK_ALPHA = 0.45

# Warna area tree
TREE_COLOR = "#FF3030"

# Warna garis batas segmentasi
CONTOUR_COLOR = "#FFFF00"


# ============================================================
# FUNGSI STRETCH RGB
# ============================================================

def stretch_rgb(image: np.ndarray) -> np.ndarray:
    """
    Mengubah citra RGB menjadi citra uint8 agar nyaman ditampilkan.

    Input:
        image: bentuk (3, tinggi, lebar)

    Output:
        RGB: bentuk (tinggi, lebar, 3)
    """

    image = np.moveaxis(
        image,
        0,
        -1
    ).astype(np.float32)

    valid = np.all(
        np.isfinite(image),
        axis=2
    )

    valid &= np.any(
        image != 0,
        axis=2
    )

    valid &= ~np.all(
        image >= 65535,
        axis=2
    )

    if not np.any(valid):
        raise RuntimeError(
            "Citra hanya berisi NoData atau nilai kosong."
        )

    preview = np.full(
        image.shape,
        255,
        dtype=np.uint8
    )

    for band_index in range(3):
        band = image[:, :, band_index]

        low, high = np.percentile(
            band[valid],
            (0.5, 99.5)
        )

        if high <= low:
            high = low + 1

        stretched = (
            (band - low)
            / (high - low)
            * 255
        )

        preview[:, :, band_index] = np.clip(
            stretched,
            0,
            255
        ).astype(np.uint8)

    preview[~valid] = 255

    return preview


# ============================================================
# FUNGSI FORMAT ANGKA
# ============================================================

def format_number(value: float) -> str:
    """
    Format angka agar lebih mudah dibaca.
    """

    return f"{value:,.0f}".replace(",", ".")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    root = Path(__file__).resolve().parent

    image_path = (
        root
        / "dataset"
        / "data"
        / "input"
        / IMAGE_NAME
    )

    mask_path = (
        root
        / "outputs"
        / MASK_NAME
    )

    probability_path = (
        root
        / "outputs"
        / PROBABILITY_NAME
    )

    preview_path = (
        root
        / "outputs"
        / f"preview_detail_resnet50_{Path(IMAGE_NAME).stem}.png"
    )

    # --------------------------------------------------------
    # Validasi file
    # --------------------------------------------------------

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Citra input tidak ditemukan:\n{image_path}"
        )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Mask hasil ResNet50 tidak ditemukan:\n{mask_path}\n"
            "Jalankan predict_resnet50.py terlebih dahulu."
        )

    # --------------------------------------------------------
    # Baca citra RGB
    # --------------------------------------------------------

    print("[INFO] Membaca citra RGB...")

    with rasterio.open(image_path) as image_src:
        scale = min(
            1.0,
            MAX_PREVIEW_PIXELS
            / max(image_src.width, image_src.height)
        )

        preview_width = max(
            1,
            round(image_src.width * scale)
        )

        preview_height = max(
            1,
            round(image_src.height * scale)
        )

        rgb = image_src.read(
            [1, 2, 3],
            out_shape=(
                3,
                preview_height,
                preview_width
            ),
            resampling=Resampling.bilinear
        )

        rgb = stretch_rgb(rgb)

        bounds = image_src.bounds
        raster_crs = image_src.crs

        original_width = image_src.width
        original_height = image_src.height

        image_transform = image_src.transform

    # --------------------------------------------------------
    # Baca mask hasil prediksi
    # --------------------------------------------------------

    print("[INFO] Membaca mask hasil ResNet50...")

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(
            1,
            out_shape=(
                preview_height,
                preview_width
            ),
            resampling=Resampling.nearest
        )

        mask_crs = mask_src.crs

        mask_transform = mask_src.transform

        mask_width = mask_src.width
        mask_height = mask_src.height

    if (
        mask_width != original_width
        or mask_height != original_height
    ):
        raise RuntimeError(
            "Ukuran mask berbeda dengan citra input.\n"
            f"Citra asli: {original_width} x {original_height}\n"
            f"Mask      : {mask_width} x {mask_height}"
        )

    # --------------------------------------------------------
    # Informasi mask
    # --------------------------------------------------------

    unique_values, unique_counts = np.unique(
        mask,
        return_counts=True
    )

    print("\n[INFO] Nilai unik pada mask:")

    for value, count in zip(
        unique_values,
        unique_counts
    ):
        print(
            f"  Nilai {value}: "
            f"{format_number(count)} piksel"
        )

    # Asumsi:
    # 0 = background
    # nilai > 0 = tree

    tree_mask = mask > 0

    background_pixels = int(
        np.count_nonzero(~tree_mask)
    )

    tree_pixels = int(
        np.count_nonzero(tree_mask)
    )

    total_pixels = int(
        tree_mask.size
    )

    tree_percentage = (
        tree_pixels
        / max(total_pixels, 1)
        * 100
    )

    background_percentage = (
        background_pixels
        / max(total_pixels, 1)
        * 100
    )

    # --------------------------------------------------------
    # Cek kesesuaian CRS
    # --------------------------------------------------------

    if raster_crs != mask_crs:
        print("\n[WARNING] CRS citra dan mask berbeda.")
        print("CRS citra:", raster_crs)
        print("CRS mask :", mask_crs)

    # --------------------------------------------------------
    # Buat mask RGB untuk tampilan
    # --------------------------------------------------------

    overlay_mask = tree_mask.astype(np.uint8)

    mask_cmap = ListedColormap(
        [
            (0, 0, 0, 0),
            TREE_COLOR
        ]
    )

    # --------------------------------------------------------
    # Buat figure 2 x 2
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 16)
    )

    ax_rgb = axes[0, 0]
    ax_mask = axes[0, 1]
    ax_overlay = axes[1, 0]
    ax_contour = axes[1, 1]

    extent = (
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    )

    # --------------------------------------------------------
    # Panel 1: Citra RGB asli
    # --------------------------------------------------------

    ax_rgb.imshow(
        rgb,
        extent=extent,
        origin="upper"
    )

    ax_rgb.set_title(
        "1. Citra RGB Asli",
        fontsize=14,
        fontweight="bold"
    )

    ax_rgb.set_axis_off()

    # --------------------------------------------------------
    # Panel 2: Mask hasil prediksi
    # --------------------------------------------------------

    ax_mask.imshow(
        overlay_mask,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(
            [
                "#FFFFFF",
                TREE_COLOR
            ]
        ),
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax_mask.set_title(
        "2. Mask Hasil ResNet50\n"
        "Putih = Background | Merah = Tree",
        fontsize=14,
        fontweight="bold"
    )

    ax_mask.set_axis_off()

    # --------------------------------------------------------
    # Panel 3: Overlay RGB + mask
    # --------------------------------------------------------

    ax_overlay.imshow(
        rgb,
        extent=extent,
        origin="upper"
    )

    ax_overlay.imshow(
        overlay_mask,
        extent=extent,
        origin="upper",
        cmap=mask_cmap,
        vmin=0,
        vmax=1,
        alpha=MASK_ALPHA,
        interpolation="nearest"
    )

    ax_overlay.set_title(
        "3. Overlay Segmentasi\n"
        f"Area Tree: {tree_percentage:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    ax_overlay.legend(
        handles=[
            Patch(
                facecolor=TREE_COLOR,
                edgecolor=TREE_COLOR,
                alpha=MASK_ALPHA,
                label="Area terprediksi sebagai tree"
            )
        ],
        loc="lower right",
        framealpha=0.9
    )

    ax_overlay.set_axis_off()

    # --------------------------------------------------------
    # Panel 4: Garis batas segmentasi
    # --------------------------------------------------------

    ax_contour.imshow(
        rgb,
        extent=extent,
        origin="upper"
    )

    # Contour hanya dibuat jika terdapat piksel tree
    if np.any(tree_mask):
        ax_contour.contour(
            tree_mask.astype(np.float32),
            levels=[0.5],
            colors=CONTOUR_COLOR,
            linewidths=0.8,
            extent=extent
        )

    ax_contour.set_title(
        "4. Batas Area Segmentasi\n"
        "Garis kuning = batas prediksi tree",
        fontsize=14,
        fontweight="bold"
    )

    ax_contour.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor=CONTOUR_COLOR,
                label="Batas segmentasi"
            )
        ],
        loc="lower right",
        framealpha=0.9
    )

    ax_contour.set_axis_off()

    # --------------------------------------------------------
    # Judul utama dan informasi
    # --------------------------------------------------------

    fig.suptitle(
        "Visualisasi Hasil Segmentasi ResNet50\n"
        f"{Path(IMAGE_NAME).stem}",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    information_text = (
        f"Ukuran asli       : "
        f"{original_width} x {original_height} piksel\n"
        f"Ukuran preview    : "
        f"{preview_width} x {preview_height} piksel\n"
        f"Total piksel      : "
        f"{format_number(total_pixels)}\n"
        f"Background        : "
        f"{format_number(background_pixels)} piksel "
        f"({background_percentage:.2f}%)\n"
        f"Tree              : "
        f"{format_number(tree_pixels)} piksel "
        f"({tree_percentage:.2f}%)\n"
        f"Nilai mask        : "
        f"{unique_values.tolist()}"
    )

    fig.text(
        0.02,
        0.015,
        information_text,
        fontsize=11,
        verticalalignment="bottom",
        family="monospace"
    )

    plt.tight_layout(
        rect=(0, 0.08, 1, 0.94)
    )

    # --------------------------------------------------------
    # Simpan file
    # --------------------------------------------------------

    fig.savefig(
        preview_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("PREVIEW DETAIL RESNET50 BERHASIL DIBUAT")
    print("=" * 70)
    print("File preview:", preview_path)
    print("Ukuran citra:", f"{original_width} x {original_height}")
    print("Background  :", f"{background_percentage:.2f}%")
    print("Tree        :", f"{tree_percentage:.2f}%")
    print("Nilai mask  :", unique_values.tolist())
    print("=" * 70)


if __name__ == "__main__":
    main()