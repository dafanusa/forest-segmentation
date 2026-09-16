from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer import swin_transformer


# ============================================================
# KONFIGURASI
# ============================================================

IMAGE_PATH = Path("data/Danum_RGB_project_to_CHM.tif")
CHECKPOINT_PATH = Path("checkpoints/swin_best.pth")
OUTPUT_DIR = Path("outputs")

OUTPUT_MASK = OUTPUT_DIR / "swin_mask.tif"

TILE_SIZE = 256
OVERLAP = 32

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL SWIN + SEGMENTATION HEAD
# ============================================================

class SwinSegmentation(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = swin_transformer()

        # Source Swin mengeluarkan 1024 channel
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feature = self.backbone(x)

        output = self.classifier(feature)

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
            f"Checkpoint tidak ditemukan: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("Checkpoint berhasil dimuat")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model


# ============================================================
# BACA GEOTIFF
# ============================================================

def read_geotiff(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile.copy()

    image = image.astype(np.float32)

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    if image.shape[0] > 3:
        image = image[:3]

    normalized = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        band = image[i]
        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        low = np.percentile(band[valid], 2)
        high = np.percentile(band[valid], 98)

        if high <= low:
            high = low + 1

        band = np.clip(band, low, high)
        normalized[i] = (band - low) / (high - low)

    return normalized, profile


def normalize_image(image):
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(3, 1, 1)

    return (image - mean) / std


# ============================================================
# INFERENSI TILE
# ============================================================

@torch.no_grad()
def predict_large_image(model, image):
    _, height, width = image.shape

    stride = TILE_SIZE - OVERLAP

    logits_sum = np.zeros(
        (NUM_CLASSES, height, width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for y in range(0, height, stride):
        for x in range(0, width, stride):

            y1 = y
            x1 = x

            y2 = min(y1 + TILE_SIZE, height)
            x2 = min(x1 + TILE_SIZE, width)

            tile = image[:, y1:y2, x1:x2]

            tile_height = tile.shape[1]
            tile_width = tile.shape[2]

            padded = np.zeros(
                (3, TILE_SIZE, TILE_SIZE),
                dtype=np.float32
            )

            padded[:, :tile_height, :tile_width] = tile

            padded = normalize_image(padded)

            tensor = torch.from_numpy(
                padded
            ).unsqueeze(0).float().to(DEVICE)

            output = model(tensor)

            output = output.squeeze(0).cpu().numpy()
            output = output[:, :tile_height, :tile_width]

            logits_sum[
                :,
                y1:y2,
                x1:x2
            ] += output

            count_map[
                y1:y2,
                x1:x2
            ] += 1

            print(
                f"Tile selesai: x={x1}, y={y1}"
            )

    logits_sum /= np.maximum(
        count_map[None, :, :],
        1
    )

    prediction = np.argmax(
        logits_sum,
        axis=0
    ).astype(np.uint8)

    return prediction


# ============================================================
# SIMPAN MASK
# ============================================================

def save_mask(mask, profile, output_path):
    profile.update(
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=255
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:
        dst.write(mask, 1)

    print(f"Mask disimpan ke: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Device:", DEVICE)
    print("Model: Swin Transformer saja")

    image, profile = read_geotiff(IMAGE_PATH)

    model = SwinSegmentation(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model = load_checkpoint(
        model,
        CHECKPOINT_PATH
    )

    model.eval()

    prediction = predict_large_image(
        model,
        image
    )

    save_mask(
        prediction,
        profile,
        OUTPUT_MASK
    )

    print("Inferensi Swin Transformer selesai")


if __name__ == "__main__":
    main()