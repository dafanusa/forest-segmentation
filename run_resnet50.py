from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet50 import resnet50


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "data"
    / "input"
    / "Dan_2014_RGB_project_to_CHM.tif"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "resnet50s-a75c83cf.pth"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_MASK = OUTPUT_DIR / "resnet50_mask.tif"

TILE_SIZE = 256
OVERLAP = 32

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL RESNET50 + SEGMENTATION HEAD
# ============================================================

class ResNet50Segmentation(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = resnet50(pretrained=False)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                2048,
                512,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):
        _, deep_feature = self.backbone(x)

        output = self.classifier(deep_feature)

        output = F.interpolate(
            output,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return output


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nCheckpoint tidak ditemukan.\n"
            f"Lokasi yang dicari:\n{checkpoint_path}\n\n"
            "Pastikan file resnet50s-a75c83cf.pth berada di folder "
            "checkpoints."
        )

    print(f"Memuat checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Format checkpoint tidak dikenali. "
            "Checkpoint harus berupa state_dict atau dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("\n========== HASIL LOAD CHECKPOINT ==========")
    print(f"Jumlah parameter checkpoint : {len(cleaned_state_dict)}")
    print(f"Jumlah missing keys         : {len(missing_keys)}")
    print(f"Jumlah unexpected keys      : {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing keys:")
        for key in missing_keys:
            print(f"  - {key}")

    if unexpected_keys:
        print("\nUnexpected keys:")
        for key in unexpected_keys:
            print(f"  - {key}")

    print("===========================================\n")

    return model


# ============================================================
# NORMALISASI TILE
# ============================================================

def normalize_tile(tile):
    """
    tile memiliki format:
    (band, height, width)

    Fungsi ini mengubah input menjadi 3 channel float32
    dengan nilai 0 sampai 1.
    """

    tile = tile.astype(np.float32, copy=False)

    # Jika hanya satu band, ulangi menjadi 3 channel.
    if tile.shape[0] == 1:
        tile = np.repeat(tile, 3, axis=0)

    # Jika lebih dari 3 band, gunakan 3 band pertama.
    if tile.shape[0] > 3:
        tile = tile[:3]

    if tile.shape[0] != 3:
        raise ValueError(
            f"Jumlah channel tidak sesuai. "
            f"Diperoleh {tile.shape[0]} channel, "
            f"seharusnya 3."
        )

    normalized = np.zeros_like(
        tile,
        dtype=np.float32
    )

    for band_index in range(3):
        band = tile[band_index]

        valid = np.isfinite(band)

        if not np.any(valid):
            print(
                f"Peringatan: channel {band_index + 1} "
                "tidak memiliki nilai valid."
            )
            continue

        low = np.percentile(
            band[valid],
            2
        )

        high = np.percentile(
            band[valid],
            98
        )

        if not np.isfinite(low):
            low = 0.0

        if not np.isfinite(high):
            high = 1.0

        if high <= low:
            high = low + 1.0

        band = np.nan_to_num(
            band,
            nan=low,
            posinf=high,
            neginf=low
        )

        band = np.clip(
            band,
            low,
            high
        )

        band = (
            band - low
        ) / (
            high - low
        )

        normalized[band_index] = band

    return normalized


# ============================================================
# NORMALISASI IMAGENET
# ============================================================

def normalize_image(image):
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(3, 1, 1)

    return (
        image - mean
    ) / std


# ============================================================
# PREDIKSI SATU TILE
# ============================================================

@torch.no_grad()
def predict_tile(model, tile):
    """
    tile:
        numpy array dengan shape (3, height, width)
    """

    original_height = tile.shape[1]
    original_width = tile.shape[2]

    padded = np.zeros(
        (3, TILE_SIZE, TILE_SIZE),
        dtype=np.float32
    )

    padded[
        :,
        :original_height,
        :original_width
    ] = tile

    padded = normalize_image(padded)

    tensor = torch.from_numpy(
        padded
    ).unsqueeze(0).float().to(DEVICE)

    output = model(tensor)

    if output.ndim != 4:
        raise RuntimeError(
            f"Output model harus memiliki 4 dimensi, "
            f"tetapi mendapatkan shape {output.shape}."
        )

    if output.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Output model memiliki {output.shape[1]} kelas, "
            f"sedangkan NUM_CLASSES={NUM_CLASSES}."
        )

    output = output.squeeze(0).cpu().numpy()

    output = output[
        :,
        :original_height,
        :original_width
    ]

    prediction = np.argmax(
        output,
        axis=0
    ).astype(np.uint8)

    return prediction


# ============================================================
# INFERENSI LANGSUNG DARI GEOTIFF KE OUTPUT
# ============================================================

def predict_geotiff(model, input_path, output_path):
    if not input_path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan:\n{input_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stride = TILE_SIZE - OVERLAP

    if stride <= 0:
        raise ValueError(
            "OVERLAP harus lebih kecil daripada TILE_SIZE."
        )

    print("\n========== INFORMASI GEOTIFF ==========")

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height

        print(f"Jumlah band : {src.count}")
        print(f"Ukuran citra: {width} x {height}")
        print(f"CRS        : {src.crs}")
        print(f"Tile size  : {TILE_SIZE}")
        print(f"Overlap    : {OVERLAP}")
        print(f"Stride     : {stride}")

        # Menghitung jumlah tile.
        y_positions = list(range(0, height, stride))
        x_positions = list(range(0, width, stride))

        total_tiles = len(y_positions) * len(x_positions)

        print(f"Total tile : {total_tiles}")
        print("=======================================\n")

        output_profile = src.profile.copy()

        output_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            compress="lzw",
            nodata=255,
            BIGTIFF="YES"
        )

        with rasterio.open(
            output_path,
            "w",
            **output_profile
        ) as dst:

            tile_number = 0

            for y in y_positions:
                for x in x_positions:
                    tile_number += 1

                    tile_width = min(
                        TILE_SIZE,
                        width - x
                    )

                    tile_height = min(
                        TILE_SIZE,
                        height - y
                    )

                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=tile_width,
                        height=tile_height
                    )

                    try:
                        tile = src.read(
                            window=window
                        )

                        tile = normalize_tile(tile)

                        prediction = predict_tile(
                            model,
                            tile
                        )

                        dst.write(
                            prediction,
                            1,
                            window=window
                        )

                        print(
                            f"[{tile_number}/{total_tiles}] "
                            f"Tile selesai: "
                            f"x={x}, y={y}, "
                            f"ukuran={tile_width}x{tile_height}"
                        )

                    except Exception as error:
                        print(
                            f"\nGAGAL membaca atau memproses tile "
                            f"x={x}, y={y}"
                        )
                        print(f"Error: {repr(error)}")
                        raise

    print("\nDistribusi kelas tidak dihitung penuh agar RAM tetap hemat.")

    print("\nMask berhasil disimpan ke:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("==========================================")
    print("       INFERENSI RESNET50 STRD-NET        ")
    print("==========================================")
    print(f"Device: {DEVICE}")
    print(f"Input : {IMAGE_PATH}")
    print("Model : ResNet50 dengan segmentation head")
    print(f"Kelas : {NUM_CLASSES}")
    print("==========================================\n")

    model = ResNet50Segmentation(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model = load_checkpoint(
        model,
        CHECKPOINT_PATH
    )

    model.eval()

    predict_geotiff(
        model,
        IMAGE_PATH,
        OUTPUT_MASK
    )

    print("\n==========================================")
    print("       INFERENSI RESNET50 SELESAI        ")
    print("==========================================")


if __name__ == "__main__":
    main()