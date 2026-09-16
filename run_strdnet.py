from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch

from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import unary_union


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "data" / "input"
MODEL_DIR = ROOT / "checkpoint"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_NAME = "Dan_2014_RGB_project_to_CHM.tif"
MODEL_NAME = "va_11011_strdnet_2.pth"
OUTPUT_NAME = "strdnet_Danum_2014.gpkg"

IMAGE_PATH = IMAGE_DIR / IMAGE_NAME
MODEL_PATH = MODEL_DIR / MODEL_NAME
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_NAME

# STRD-Net menggunakan input 256 x 256
TILE_SIZE = 256

# Overlap antar-tile
STRIDE = 192

# Threshold prediksi
SCORE_THRESHOLD = 0.50

# Luas minimum polygon dalam satuan CRS citra
# CRS EPSG:32650 menggunakan meter, sehingga luasnya m2
MIN_AREA_M2 = 2.0

# Jika True, polygon yang saling bersinggungan akan digabung.
# Untuk mempertahankan polygon pohon secara terpisah, gunakan False.
MERGE_TOUCHING_POLYGONS = False

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# IMPORT ARSITEKTUR STRD-NET
# ============================================================

try:
    from model.STRD_Net import STRD_Net
except Exception as error:
    print("Gagal mengimpor arsitektur STRD-Net.")
    print("Pastikan file berikut tersedia:")
    print("model/STRD_Net.py")
    print()
    print("Detail error:")
    print(error)
    sys.exit(1)


# ============================================================
# UTILITAS
# ============================================================

def check_required_files() -> None:
    """Memeriksa file input dan checkpoint model."""

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Citra input tidak ditemukan:\n{IMAGE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model STRD-Net tidak ditemukan:\n{MODEL_PATH}\n\n"
            "Pastikan file model berada di folder checkpoint."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalisasi citra menjadi float32 dengan rentang 0 sampai 1.

    Input:
        H x W x C

    Output:
        H x W x C
    """

    image = image.astype(np.float32)

    # Menangani NaN dan infinite
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    # Jika citra belum berada pada rentang 0-1,
    # lakukan normalisasi min-max pada tile.
    if max_value > 1.0 or min_value < 0.0:
        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        else:
            image = np.zeros_like(image)

    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


def pad_to_model_size(
    image: np.ndarray,
    target_size: int = 256
) -> tuple[np.ndarray, int, int]:
    """
    Padding tile menjadi target_size x target_size.

    Input:
        H x W x C

    Output:
        padded_image : target_size x target_size x C
        original_h   : tinggi asli
        original_w   : lebar asli
    """

    original_h, original_w, channels = image.shape

    padded_image = np.zeros(
        (target_size, target_size, channels),
        dtype=image.dtype
    )

    copy_h = min(original_h, target_size)
    copy_w = min(original_w, target_size)

    padded_image[
        :copy_h,
        :copy_w,
        :
    ] = image[
        :copy_h,
        :copy_w,
        :
    ]

    return padded_image, original_h, original_w


def load_model(
    model_path: Path,
    device: torch.device
) -> torch.nn.Module:
    """
    Memuat checkpoint STRD-Net.

    Checkpoint dapat berupa:
    1. model utuh torch.nn.Module
    2. state_dict
    3. dictionary yang memiliki key state_dict/model_state_dict
    """

    print("[1/5] Memuat arsitektur STRD-Net...")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = STRD_Net()

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint

    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

        # Menghapus prefix "module." jika model dilatih
        # menggunakan DataParallel.
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned_key = key[len("module."):]
            else:
                cleaned_key = key

            cleaned_state_dict[cleaned_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        if missing_keys:
            print("Peringatan: terdapat parameter model yang tidak termuat.")
            print("Missing keys:", missing_keys[:10])

        if unexpected_keys:
            print("Peringatan: terdapat parameter tambahan pada checkpoint.")
            print("Unexpected keys:", unexpected_keys[:10])

    else:
        raise TypeError(
            "Format checkpoint tidak dikenali. "
            "Checkpoint harus berupa torch.nn.Module atau state_dict."
        )

    model = model.to(device)
    model.eval()

    print("Model STRD-Net berhasil dimuat.")
    print(f"Device: {device}")

    return model


def prepare_tensor(
    image: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Mengubah citra HWC menjadi tensor BCHW.
    """

    image = normalize_image(image)

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()

    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def extract_prediction_tensor(
    output: torch.Tensor | tuple | list
) -> torch.Tensor:
    """
    Mengambil tensor prediksi dari output model.

    Output model dapat berupa tensor, tuple, atau list.
    """

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise RuntimeError("Output model berupa tuple/list kosong.")

        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            f"Output model bukan tensor: {type(output)}"
        )

    return output


def predict_tile(
    model: torch.nn.Module,
    rgb_tile: np.ndarray,
    second_tile: np.ndarray,
    device: torch.device,
    tile_size: int = 256,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Melakukan prediksi pada satu tile.

    Tile yang berukuran kurang dari 256x256 akan dipadding.
    Setelah prediksi, area padding dibuang kembali.
    """

    rgb_padded, original_h, original_w = pad_to_model_size(
        rgb_tile,
        target_size=tile_size
    )

    second_padded, _, _ = pad_to_model_size(
        second_tile,
        target_size=tile_size
    )

    rgb_tensor = prepare_tensor(
        rgb_padded,
        device
    )

    second_tensor = prepare_tensor(
        second_padded,
        device
    )

    with torch.no_grad():
        output = model(
            rgb_tensor,
            second_tensor
        )

    output = extract_prediction_tensor(output)

    # Bentuk umum output:
    # B x C x H x W
    if output.ndim == 4:
        prediction = output[0]

        # Jika memiliki lebih dari satu channel,
        # gunakan channel terakhir.
        if prediction.shape[0] > 1:
            prediction = prediction[-1]
        else:
            prediction = prediction[0]

    # Bentuk alternatif:
    # B x H x W
    elif output.ndim == 3:
        prediction = output[0]

    # Bentuk alternatif:
    # H x W
    elif output.ndim == 2:
        prediction = output

    else:
        raise RuntimeError(
            f"Bentuk output model tidak dikenali: {tuple(output.shape)}"
        )

    # Jika output belum berupa probabilitas,
    # ubah menggunakan sigmoid.
    prediction = torch.sigmoid(prediction)

    prediction = prediction.detach().cpu().numpy()

    # Buang area padding.
    prediction = prediction[
        :original_h,
        :original_w
    ]

    binary_mask = prediction >= threshold

    return binary_mask.astype(np.uint8)


def mask_to_polygons(
    mask: np.ndarray,
    transform,
    min_area: float = 2.0
) -> list:
    """
    Mengubah mask biner menjadi polygon.
    """

    polygons = []

    for geometry, value in shapes(
        mask,
        mask=(mask > 0),
        transform=transform
    ):
        if value != 1:
            continue

        polygon = shape(geometry)

        if polygon.is_empty:
            continue

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        if polygon.area < min_area:
            continue

        polygons.append(polygon)

    return polygons


def remove_duplicate_polygons(
    polygons: list
) -> list:
    """
    Menghapus polygon yang identik atau sangat beririsan.
    Fungsi ini sederhana dan digunakan untuk mengurangi
    duplikasi polygon dari area overlap tile.
    """

    if not polygons:
        return []

    filtered = []

    for polygon in polygons:
        duplicate = False

        for existing in filtered:
            intersection_area = polygon.intersection(existing).area
            smaller_area = min(
                polygon.area,
                existing.area
            )

            if smaller_area > 0:
                overlap_ratio = intersection_area / smaller_area

                if overlap_ratio >= 0.80:
                    duplicate = True
                    break

        if not duplicate:
            filtered.append(polygon)

    return filtered


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PREDIKSI TAJUK POHON MENGGUNAKAN STRD-NET")
    print("=" * 70)

    check_required_files()

    device = torch.device(DEVICE)

    print(f"Citra : {IMAGE_PATH}")
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {device}")

    model = load_model(
        model_path=MODEL_PATH,
        device=device
    )

    print()
    print("[2/5] Membaca metadata citra...")

    with rasterio.open(IMAGE_PATH) as src:
        width = src.width
        height = src.height
        band_count = src.count
        crs = src.crs
        transform = src.transform
        resolution = src.res
        dtype = src.dtypes
        nodata = src.nodata

        print(f"CRS         : {crs}")
        print(f"Jumlah band : {band_count}")
        print(f"Ukuran      : {width} x {height}")
        print(f"Resolusi    : {resolution}")
        print(f"Tipe data   : {dtype}")
        print(f"NoData      : {nodata}")

        if band_count < 3:
            raise ValueError(
                "Citra harus memiliki minimal 3 band RGB."
            )

        print()
        print("[3/5] Menjalankan prediksi tile...")

        all_polygons = []
        tile_counter = 0

        for y in range(0, height, STRIDE):
            for x in range(0, width, STRIDE):

                tile_width = min(
                    TILE_SIZE,
                    width - x
                )

                tile_height = min(
                    TILE_SIZE,
                    height - y
                )

                if tile_width <= 0 or tile_height <= 0:
                    continue

                window = Window(
                    col_off=x,
                    row_off=y,
                    width=tile_width,
                    height=tile_height
                )

                # Membaca RGB
                rgb_tile = src.read(
                    [1, 2, 3],
                    window=window
                ).transpose(1, 2, 0)

                # ====================================================
                # INPUT KEDUA
                # ====================================================
                #
                # STRD-Net membutuhkan input kedua berupa NDVI.
                #
                # Karena file yang digunakan adalah RGB + CHM,
                # tidak boleh menganggap band ke-4 sebagai NIR.
                #
                # Untuk sementara, RGB digunakan sebagai input kedua
                # agar model dapat dijalankan secara teknis.
                #
                # Untuk hasil ilmiah yang benar, ganti second_tile
                # dengan citra NDVI yang sesuai dengan dataset training.
                # ====================================================

                second_tile = rgb_tile.copy()

                prediction = predict_tile(
                    model=model,
                    rgb_tile=rgb_tile,
                    second_tile=second_tile,
                    device=device,
                    tile_size=TILE_SIZE,
                    threshold=SCORE_THRESHOLD
                )

                tile_transform = rasterio.windows.transform(
                    window,
                    transform
                )

                polygons = mask_to_polygons(
                    mask=prediction,
                    transform=tile_transform,
                    min_area=MIN_AREA_M2
                )

                all_polygons.extend(polygons)

                tile_counter += 1

                if tile_counter % 10 == 0:
                    print(
                        f"Tile diproses: {tile_counter}, "
                        f"polygon sementara: {len(all_polygons)}"
                    )

    print()
    print("[4/5] Membersihkan polygon hasil prediksi...")

    all_polygons = [
        polygon
        for polygon in all_polygons
        if polygon is not None
        and not polygon.is_empty
        and polygon.is_valid
        and polygon.area >= MIN_AREA_M2
    ]

    print(
        f"Jumlah polygon sebelum penghapusan duplikasi: "
        f"{len(all_polygons)}"
    )

    all_polygons = remove_duplicate_polygons(
        all_polygons
    )

    if MERGE_TOUCHING_POLYGONS and all_polygons:
        merged = unary_union(all_polygons)

        if merged.geom_type == "Polygon":
            all_polygons = [merged]

        elif merged.geom_type == "MultiPolygon":
            all_polygons = list(merged.geoms)

    print(
        f"Jumlah polygon setelah pembersihan: "
        f"{len(all_polygons)}"
    )

    print()
    print("[5/5] Menyimpan hasil ke GeoPackage...")

    if not all_polygons:
        print("Tidak ada polygon yang dihasilkan.")
        return

    result = gpd.GeoDataFrame(
        {
            "tree_id": range(1, len(all_polygons) + 1),
            "area_m2": [
                polygon.area
                for polygon in all_polygons
            ],
            "geometry": all_polygons
        },
        crs=crs
    )

    result.to_file(
        OUTPUT_PATH,
        layer="tree_crowns",
        driver="GPKG"
    )

    print()
    print("=" * 70)
    print("PREDIKSI SELESAI")
    print("=" * 70)
    print(f"Jumlah polygon : {len(result)}")
    print(f"Output         : {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()