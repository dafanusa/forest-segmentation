from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from matplotlib.lines import Line2D
from rasterio.enums import Resampling


# ============================================================
# KONFIGURASI
# ============================================================

IMAGE_NAME = "Dan_2014_RGB_project_to_CHM.tif"

PREDICTION_NAME = "crowns_Danum_resnet50.gpkg"

# Isi None jika belum memiliki ground truth
REFERENCE_NAME = None

# Contoh jika ground truth tersedia:
# REFERENCE_NAME = "Danum.gpkg"

MAX_PREVIEW_PIXELS = 2400


# ============================================================
# STRETCH RGB
# ============================================================

def stretch_rgb(image: np.ndarray) -> np.ndarray:
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
            "Citra hanya berisi NoData."
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
# MAIN
# ============================================================

def main() -> None:
    root = Path(__file__).resolve().parent

    image_path = (
        root
        / "data"
        / "input"
        / IMAGE_NAME
    )

    prediction_path = (
        root
        / "outputs"
        / PREDICTION_NAME
    )

    preview_path = (
        root
        / "outputs"
        / f"preview_polygon_resnet50_{Path(IMAGE_NAME).stem}.png"
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Citra tidak ditemukan:\n{image_path}"
        )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Polygon prediksi tidak ditemukan:\n{prediction_path}\n"
            "Jalankan mask_to_polygon_resnet50.py terlebih dahulu."
        )

    # --------------------------------------------------------
    # Baca citra
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:
        scale = min(
            1.0,
            MAX_PREVIEW_PIXELS
            / max(src.width, src.height)
        )

        preview_width = max(
            1,
            round(src.width * scale)
        )

        preview_height = max(
            1,
            round(src.height * scale)
        )

        rgb = src.read(
            [1, 2, 3],
            out_shape=(
                3,
                preview_height,
                preview_width
            ),
            resampling=Resampling.bilinear
        )

        rgb = stretch_rgb(rgb)

        bounds = src.bounds
        raster_crs = src.crs

    # --------------------------------------------------------
    # Baca polygon prediksi
    # --------------------------------------------------------

    predictions = gpd.read_file(
        prediction_path
    )

    if predictions.empty:
        raise RuntimeError(
            "File polygon prediksi kosong."
        )

    if raster_crs and predictions.crs != raster_crs:
        predictions = predictions.to_crs(
            raster_crs
        )

    predictions = predictions.cx[
        bounds.left:bounds.right,
        bounds.bottom:bounds.top,
    ]

    # --------------------------------------------------------
    # Baca ground truth jika tersedia
    # --------------------------------------------------------

    references = None

    if REFERENCE_NAME is not None:
        reference_path = (
            root
            / "data"
            / "reference"
            / REFERENCE_NAME
        )

        if reference_path.is_file():
            references = gpd.read_file(
                reference_path
            )

            if (
                raster_crs
                and references.crs != raster_crs
            ):
                references = references.to_crs(
                    raster_crs
                )

            references = references.cx[
                bounds.left:bounds.right,
                bounds.bottom:bounds.top,
            ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    ax.imshow(
        rgb,
        extent=(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ),
        origin="upper"
    )

    legend_items = []

    # Ground truth cyan
    if references is not None and not references.empty:
        references.boundary.plot(
            ax=ax,
            color="#00FFFF",
            linewidth=0.8,
            alpha=0.9
        )

        legend_items.append(
            Line2D(
                [0],
                [0],
                color="#00FFFF",
                linewidth=2,
                label="Ground truth"
            )
        )

    # Prediksi ResNet50 merah
    predictions.boundary.plot(
        ax=ax,
        color="#FF3030",
        linewidth=0.8,
        alpha=0.95
    )

    legend_items.append(
        Line2D(
            [0],
            [0],
            color="#FF3030",
            linewidth=2,
            label="Prediksi ResNet50"
        )
    )

    ax.set_xlim(
        bounds.left,
        bounds.right
    )

    ax.set_ylim(
        bounds.bottom,
        bounds.top
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.set_title(
        f"Delineasi Tajuk Pohon – "
        f"{Path(IMAGE_NAME).stem}\n"
        f"Prediksi ResNet50: "
        f"{len(predictions)} polygon",
        fontsize=14,
        pad=12
    )

    ax.legend(
        handles=legend_items,
        loc="lower right",
        framealpha=0.9
    )

    fig.savefig(
        preview_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("=" * 65)
    print("PREVIEW POLYGON RESNET50 BERHASIL")
    print("=" * 65)
    print("Jumlah polygon :", len(predictions))
    print("File preview   :", preview_path)
    print("=" * 65)


if __name__ == "__main__":
    main()