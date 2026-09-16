from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet50 import resnet50


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
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

OUTPUT_MASK_PATH = OUTPUT_DIR / "Danum_resnet50_mask.tif"
OUTPUT_PROBABILITY_PATH = OUTPUT_DIR / "Danum_resnet50_probability.tif"

# Sesuaikan dengan jumlah kelas pada dataset.
# 2 kelas:
#   0 = background
#   1 = tree
#
# Jika dataset memiliki background, tree, dan kelas lain,
# ubah menjadi 3 atau sesuai jumlah kelas sebenarnya.
NUM_CLASSES = 2

# Ukuran tile agar citra besar tidak langsung masuk ke GPU/RAM
TILE_SIZE = 512

# Overlap antartile untuk mengurangi garis sambungan
OVERLAP = 64

# Threshold probabilitas kelas tree
TREE_THRESHOLD = 0.50

# Gunakan CPU jika tidak memiliki CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL SEGMENTASI BERBASIS RESNET50
# ============================================================

class ResNet50Segmentation(nn.Module):
    """
    Backbone ResNet50 menghasilkan feature map dengan
    2048 channel, kemudian diteruskan ke segmentation head.

    Catatan:
    Checkpoint resnet50s-a75c83cf.pth hanya berisi bobot
    backbone ResNet50, bukan bobot segmentation head.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.backbone = resnet50(
            pretrained=False,
            num_classes=1000
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Source ResNet50 yang digunakan mengembalikan:
        # low_feature, deep_feature
        backbone_output = self.backbone(x)

        if isinstance(backbone_output, (tuple, list)):
            deep_feature = backbone_output[-1]
        else:
            deep_feature = backbone_output

        logits = self.classifier(deep_feature)

        # Kembalikan ukuran output agar sama dengan input
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device
) -> nn.Module:
    """
    Memuat checkpoint ResNet50.

    Checkpoint dapat memiliki format:
    - state_dict langsung
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": ...}

    Jika key checkpoint tidak memiliki prefix 'backbone.',
    prefix tersebut akan ditambahkan secara otomatis.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan:\n{checkpoint_path}"
        )

    print(f"[INFO] Membaca checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Ambil state_dict dari beberapa kemungkinan format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Format checkpoint tidak dikenali. "
            "Checkpoint harus berupa dictionary state_dict."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Hilangkan prefix DataParallel
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Jika checkpoint hanya berisi bobot ResNet50,
        # tambahkan prefix backbone.
        if (
            not new_key.startswith("backbone.")
            and not new_key.startswith("classifier.")
        ):
            new_key = f"backbone.{new_key}"

        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("\n[INFO] Hasil pemuatan checkpoint")

    if missing_keys:
        print(f"[WARNING] Missing keys: {len(missing_keys)}")
        print("Contoh missing keys:")
        for key in missing_keys[:10]:
            print(f"  - {key}")

    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {len(unexpected_keys)}")
        print("Contoh unexpected keys:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")

    if not missing_keys and not unexpected_keys:
        print("[INFO] Semua bobot berhasil dimuat.")

    if any(key.startswith("classifier.") for key in missing_keys):
        print(
            "\n[WARNING] Bobot classifier segmentation belum tersedia. "
            "Segmentation head masih menggunakan bobot acak."
        )

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_image(image: np.ndarray) -> torch.Tensor:
    """
    Mengubah citra RGB menjadi tensor dengan normalisasi ImageNet.

    Input:
        image: H x W x 3, tipe float32

    Output:
        tensor: 1 x 3 x H x W
    """

    image = image.astype(np.float32)

    # Jika data berada pada rentang 0-255
    if image.max() > 1.0:
        image = image / 255.0

    # Batasi nilai agar stabil
    image = np.clip(image, 0.0, 1.0)

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).float()

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


# ============================================================
# PREDIKSI TILE
# ============================================================

@torch.no_grad()
def predict_tile(
    model: nn.Module,
    tile: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Melakukan prediksi pada satu tile.

    Input:
        tile: H x W x 3

    Output:
        probability: H x W
    """

    original_height, original_width = tile.shape[:2]

    # Padding jika ukuran tile kurang dari ukuran minimum
    padded_tile = np.zeros(
        (TILE_SIZE, TILE_SIZE, 3),
        dtype=np.float32
    )

    padded_tile[:original_height, :original_width] = tile

    input_tensor = normalize_image(padded_tile).to(device)

    logits = model(input_tensor)

    probabilities = torch.softmax(logits, dim=1)

    # Kelas tree diasumsikan kelas index 1
    if NUM_CLASSES < 2:
        raise ValueError(
            "NUM_CLASSES harus minimal 2 untuk background dan tree."
        )

    tree_probability = probabilities[:, 1, :, :]

    tree_probability = tree_probability.squeeze(0).cpu().numpy()

    # Potong kembali ke ukuran tile asli
    tree_probability = tree_probability[
        :original_height,
        :original_width
    ]

    return tree_probability


# ============================================================
# PREDIKSI SELURUH CITRA
# ============================================================

def predict_raster(
    model: nn.Module,
    input_path: Path,
    output_mask_path: Path,
    output_probability_path: Path,
    device: torch.device
) -> None:

    if not input_path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan:\n{input_path}"
        )

    output_mask_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"[INFO] Membuka citra: {input_path}")

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        count = src.count
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

        print(f"[INFO] Ukuran citra : {width} x {height}")
        print(f"[INFO] Jumlah band : {count}")
        print(f"[INFO] CRS          : {crs}")

        if count < 3:
            raise ValueError(
                "Citra harus memiliki minimal 3 band untuk RGB."
            )

        # Output probability float32
        probability_profile = profile.copy()
        probability_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=0
        )

        # Output mask uint8
        mask_profile = profile.copy()
        mask_profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            compress="lzw",
            nodata=0
        )

        with rasterio.open(
            output_probability_path,
            "w",
            **probability_profile
        ) as probability_dst, rasterio.open(
            output_mask_path,
            "w",
            **mask_profile
        ) as mask_dst:

            step = TILE_SIZE - OVERLAP

            total_tiles = (
                ((height - 1) // step + 1)
                * ((width - 1) // step + 1)
            )

            tile_number = 0

            for top in range(0, height, step):
                for left in range(0, width, step):

                    bottom = min(top + TILE_SIZE, height)
                    right = min(left + TILE_SIZE, width)

                    tile_height = bottom - top
                    tile_width = right - left

                    window = rasterio.windows.Window(
                        col_off=left,
                        row_off=top,
                        width=tile_width,
                        height=tile_height
                    )

                    # Mengambil tiga band pertama sebagai RGB
                    tile = src.read(
                        indexes=[1, 2, 3],
                        window=window
                    )

                    # CHW -> HWC
                    tile = np.transpose(tile, (1, 2, 0))

                    # Konversi ke float32
                    tile = tile.astype(np.float32)

                    probability = predict_tile(
                        model=model,
                        tile=tile,
                        device=device
                    )

                    mask = (
                        probability >= TREE_THRESHOLD
                    ).astype(np.uint8)

                    probability_dst.write(
                        probability.astype(np.float32),
                        1,
                        window=window
                    )

                    mask_dst.write(
                        mask,
                        1,
                        window=window
                    )

                    tile_number += 1

                    print(
                        f"[INFO] Tile {tile_number}/{total_tiles} "
                        f"selesai"
                    )

    print("\n[INFO] Prediksi selesai.")
    print(f"[INFO] Mask       : {output_mask_path}")
    print(f"[INFO] Probability: {output_probability_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 65)
    print("PREDIKSI SEGMENTASI RESNET50")
    print("=" * 65)

    print(f"[INFO] Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "[WARNING] CUDA tidak tersedia. "
            "Prediksi akan menggunakan CPU."
        )

    print(f"[INFO] Input checkpoint: {CHECKPOINT_PATH}")
    print(f"[INFO] Input raster    : {INPUT_PATH}")
    print(f"[INFO] Jumlah kelas    : {NUM_CLASSES}")

    model = ResNet50Segmentation(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model = load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE
    )

    model.eval()

    predict_raster(
        model=model,
        input_path=INPUT_PATH,
        output_mask_path=OUTPUT_MASK_PATH,
        output_probability_path=OUTPUT_PROBABILITY_PATH,
        device=DEVICE
    )

    print("\n" + "=" * 65)
    print("PROSES BERHASIL")
    print("=" * 65)


if __name__ == "__main__":
    main()