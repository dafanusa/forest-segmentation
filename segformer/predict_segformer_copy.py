from pathlib import Path
import csv
import gc

import cv2
import numpy as np
import rasterio
import torch

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from config import (
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    TREE_THRESHOLD,
    MIN_INSTANCE_AREA,
    WATERSHED_MIN_DISTANCE,
    DEVICE,
    SEMANTIC_OUTPUT_DIR,
    INSTANCE_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR,
    create_directories,
)


# ============================================================
# KONFIGURASI FOLDER INPUT
# ============================================================

# GANTI PATH INI SESUAI LOKASI FOLDER .TIF ABANG
INPUT_TIF_DIR = Path(
    r"D:\MAHASISWA\SEMESTER 5\FUNGSIONAL\PRAKTIKUM\STRD-Net\tcd_dataset\dataset"
)

# Folder tambahan untuk menyimpan hasil detail
DETAIL_OUTPUT_DIR = Path("outputs/detail")
STATISTICS_OUTPUT_DIR = Path("outputs/statistics")

MODEL_PATH = CHECKPOINT_DIR / "best_model"


# ============================================================
# MEMBUAT FOLDER OUTPUT
# ============================================================

def create_all_directories():

    create_directories()

    SEMANTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INSTANCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OVERLAY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DETAIL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MEMBACA CITRA TIFF
# ============================================================

def read_tif(path):

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

    if image.shape[2] > 3:

        image = image[:, :, :3]

    image = image.astype(np.float32)

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            (image - image_min)
            / (image_max - image_min)
            * 255.0
        )

    return image.astype(np.uint8)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Memuat processor...")
    processor = SegformerImageProcessor.from_pretrained(
        MODEL_PATH
    )

    print("Memuat model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model berhasil dimuat pada device: {DEVICE}")

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

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
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

        tree_probability = probabilities[:, 1]

        semantic = (
            tree_probability > TREE_THRESHOLD
        ).cpu().numpy()[0].astype(np.uint8)

    return semantic


# ============================================================
# PREDIKSI SLIDING WINDOW
# ============================================================

def predict_sliding_window(
    image,
    processor,
    model,
    patch_size=512,
    stride=384,
):

    height, width = image.shape[:2]

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    total_patch = (
        ((height - 1) // stride) + 1
    ) * (
        ((width - 1) // stride) + 1
    )

    current_patch = 0

    for y in range(0, height, stride):

        for x in range(0, width, stride):

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

            ph, pw = patch.shape[:2]

            if ph != patch_size or pw != patch_size:

                padded = np.zeros(
                    (
                        patch_size,
                        patch_size,
                        3,
                    ),
                    dtype=np.uint8,
                )

                padded[:ph, :pw] = patch

                patch_input = padded

            else:

                patch_input = patch

            semantic_patch = predict_patch(
                patch_input,
                processor,
                model,
            )

            semantic_patch = semantic_patch[
                :ph,
                :pw,
            ]

            prediction_sum[
                y:y2,
                x:x2,
            ] += semantic_patch

            prediction_count[
                y:y2,
                x:x2,
            ] += 1

            if current_patch % 20 == 0 or current_patch == total_patch:

                print(
                    f"  Patch {current_patch}/{total_patch}",
                    end="\r",
                )

    semantic_mask = (
        prediction_sum
        / np.maximum(
            prediction_count,
            1,
        )
        > 0.5
    ).astype(np.uint8)

    print()

    return semantic_mask


# ============================================================
# SEMANTIC MASK MENJADI INSTANCE MASK
# ============================================================

def semantic_to_instance(semantic_mask):

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
        binary_mask
    )

    for label_id in range(1, num_labels):

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

    for idx, (row, col) in enumerate(
        coordinates,
        start=1,
    ):

        markers[row, col] = idx

    if len(coordinates) == 0:

        markers[cleaned_mask > 0] = 1

    else:

        markers = ndimage.label(
            markers > 0
        )[0]

    instance_mask = watershed(
        -distance,
        markers,
        mask=cleaned_mask,
    )

    return instance_mask.astype(np.int32)


# ============================================================
# MEMBUAT OVERLAY DETAIL
# ============================================================

def create_overlay(
    image,
    semantic_mask,
    instance_mask,
):

    overlay = image.copy()

    # Area pohon diberi warna hijau transparan
    tree_area = semantic_mask > 0

    overlay[tree_area] = (
        0.5 * overlay[tree_area]
        + 0.5 * np.array(
            [0, 255, 0]
        )
    ).astype(np.uint8)

    # Gambar boundary setiap instance/pohon
    for instance_id in np.unique(instance_mask):

        if instance_id == 0:
            continue

        binary = (
            instance_mask == instance_id
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 0, 0),
            2,
        )

    return overlay


# ============================================================
# MEMBUAT PREVIEW SEMANTIC BERWARNA
# ============================================================

def create_semantic_preview(semantic_mask):

    preview = np.zeros(
        (
            semantic_mask.shape[0],
            semantic_mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # Background berwarna hitam
    preview[semantic_mask == 0] = [
        0,
        0,
        0,
    ]

    # Area pohon berwarna hijau
    preview[semantic_mask > 0] = [
        0,
        255,
        0,
    ]

    return preview


# ============================================================
# MEMBUAT PREVIEW INSTANCE BERWARNA
# ============================================================

def create_instance_preview(instance_mask):

    height, width = instance_mask.shape

    preview = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    unique_instances = np.unique(
        instance_mask
    )

    rng = np.random.default_rng(42)

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
# MENGHITUNG STATISTIK PREDIKSI
# ============================================================

def calculate_statistics(
    image,
    semantic_mask,
    instance_mask,
):

    height, width = semantic_mask.shape

    total_pixels = height * width

    tree_pixels = int(
        np.sum(semantic_mask > 0)
    )

    background_pixels = (
        total_pixels - tree_pixels
    )

    tree_percentage = (
        tree_pixels / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    background_percentage = (
        background_pixels / total_pixels * 100
        if total_pixels > 0
        else 0
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
                instance_mask == instance_id
            )
        )

        instance_areas.append(area)

    if len(instance_areas) > 0:

        min_instance_area = min(
            instance_areas
        )

        max_instance_area = max(
            instance_areas
        )

        mean_instance_area = np.mean(
            instance_areas
        )

        median_instance_area = np.median(
            instance_areas
        )

    else:

        min_instance_area = 0
        max_instance_area = 0
        mean_instance_area = 0
        median_instance_area = 0

    statistics = {
        "height": height,
        "width": width,
        "total_pixels": total_pixels,
        "tree_pixels": tree_pixels,
        "background_pixels": background_pixels,
        "tree_percentage": tree_percentage,
        "background_percentage": background_percentage,
        "instance_count": instance_count,
        "min_instance_area_pixel": min_instance_area,
        "max_instance_area_pixel": max_instance_area,
        "mean_instance_area_pixel": mean_instance_area,
        "median_instance_area_pixel": median_instance_area,
    }

    return statistics


# ============================================================
# MENYIMPAN RINGKASAN TXT
# ============================================================

def save_detail_report(
    image_path,
    statistics,
    semantic_path,
    instance_path,
    overlay_path,
    semantic_preview_path,
    instance_preview_path,
    report_path,
):

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 75 + "\n")
        file.write("LAPORAN DETAIL HASIL INFERENCE SEGFORMER-B5\n")
        file.write("=" * 75 + "\n\n")

        file.write("INFORMASI CITRA\n")
        file.write("-" * 75 + "\n")
        file.write(f"Nama file       : {image_path.name}\n")
        file.write(f"Lokasi input    : {image_path}\n")
        file.write(
            f"Ukuran citra    : "
            f"{statistics['width']} x "
            f"{statistics['height']} piksel\n"
        )
        file.write(
            f"Total piksel    : "
            f"{statistics['total_pixels']:,}\n"
        )

        file.write("\nHASIL SEMANTIC SEGMENTATION\n")
        file.write("-" * 75 + "\n")
        file.write(
            f"Piksel pohon    : "
            f"{statistics['tree_pixels']:,}\n"
        )
        file.write(
            f"Piksel background: "
            f"{statistics['background_pixels']:,}\n"
        )
        file.write(
            f"Persentase pohon: "
            f"{statistics['tree_percentage']:.4f}%\n"
        )
        file.write(
            f"Persentase background: "
            f"{statistics['background_percentage']:.4f}%\n"
        )

        file.write("\nHASIL INSTANCE SEGMENTATION\n")
        file.write("-" * 75 + "\n")
        file.write(
            f"Jumlah instance : "
            f"{statistics['instance_count']:,}\n"
        )
        file.write(
            f"Luas instance minimum: "
            f"{statistics['min_instance_area_pixel']:,} piksel\n"
        )
        file.write(
            f"Luas instance maksimum: "
            f"{statistics['max_instance_area_pixel']:,} piksel\n"
        )
        file.write(
            f"Rata-rata luas instance: "
            f"{statistics['mean_instance_area_pixel']:.2f} piksel\n"
        )
        file.write(
            f"Median luas instance: "
            f"{statistics['median_instance_area_pixel']:.2f} piksel\n"
        )

        file.write("\nFILE OUTPUT\n")
        file.write("-" * 75 + "\n")
        file.write(f"Semantic mask       : {semantic_path}\n")
        file.write(f"Instance mask       : {instance_path}\n")
        file.write(f"Overlay             : {overlay_path}\n")
        file.write(
            f"Semantic preview    : "
            f"{semantic_preview_path}\n"
        )
        file.write(
            f"Instance preview    : "
            f"{instance_preview_path}\n"
        )

        file.write("\n" + "=" * 75 + "\n")
        file.write("SELESAI\n")
        file.write("=" * 75 + "\n")


# ============================================================
# MENYIMPAN STATISTIK CSV
# ============================================================

def save_statistics_csv(
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
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN BATCH INFERENCE
# ============================================================

def main():

    create_all_directories()

    if not INPUT_TIF_DIR.exists():

        raise FileNotFoundError(
            f"Folder input tidak ditemukan: {INPUT_TIF_DIR}"
        )

    image_paths = sorted(
        list(INPUT_TIF_DIR.glob("*.tif")) +
        list(INPUT_TIF_DIR.glob("*.tiff"))
    )

    if len(image_paths) == 0:

        raise FileNotFoundError(
            f"Tidak ditemukan file .tif atau .tiff "
            f"di folder: {INPUT_TIF_DIR}"
        )

    processor, model = load_model()

    print("=" * 75)
    print("BATCH INFERENCE SEGFORMER-B5")
    print("=" * 75)
    print(f"Folder input : {INPUT_TIF_DIR}")
    print(f"Jumlah citra : {len(image_paths)}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Device       : {DEVICE}")
    print("=" * 75)

    all_statistics = []

    berhasil = 0
    gagal = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print("-" * 75)
        print(
            f"[{index}/{len(image_paths)}] "
            f"Memproses {image_path.name}"
        )
        print("-" * 75)

        try:

            image = read_tif(
                image_path
            )

            print(
                f"Ukuran citra: "
                f"{image.shape[1]} x "
                f"{image.shape[0]}"
            )

            semantic_mask = predict_sliding_window(
                image,
                processor,
                model,
                patch_size=IMAGE_SIZE,
                stride=384,
            )

            instance_mask = semantic_to_instance(
                semantic_mask
            )

            overlay = create_overlay(
                image,
                semantic_mask,
                instance_mask,
            )

            semantic_preview = create_semantic_preview(
                semantic_mask
            )

            instance_preview = create_instance_preview(
                instance_mask
            )

            statistics = calculate_statistics(
                image,
                semantic_mask,
                instance_mask,
            )

            stem = image_path.stem

            semantic_path = (
                SEMANTIC_OUTPUT_DIR
                / f"{stem}_semantic.png"
            )

            instance_path = (
                INSTANCE_OUTPUT_DIR
                / f"{stem}_instance.tif"
            )

            overlay_path = (
                OVERLAY_OUTPUT_DIR
                / f"{stem}_overlay.png"
            )

            semantic_preview_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_semantic_preview.png"
            )

            instance_preview_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_instance_preview.png"
            )

            report_path = (
                DETAIL_OUTPUT_DIR
                / f"{stem}_report.txt"
            )

            cv2.imwrite(
                str(semantic_path),
                (semantic_mask * 255).astype(
                    np.uint8
                ),
            )

            cv2.imwrite(
                str(instance_path),
                instance_mask.astype(
                    np.int32
                ),
        )

            cv2.imwrite(
                str(overlay_path),
                cv2.cvtColor(
                    overlay,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            cv2.imwrite(
                str(semantic_preview_path),
                cv2.cvtColor(
                    semantic_preview,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            cv2.imwrite(
                str(instance_preview_path),
                cv2.cvtColor(
                    instance_preview,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            save_detail_report(
                image_path,
                statistics,
                semantic_path,
                instance_path,
                overlay_path,
                semantic_preview_path,
                instance_preview_path,
                report_path,
            )

            row = {
                "file": image_path.name,
                "width": statistics["width"],
                "height": statistics["height"],
                "total_pixels": statistics["total_pixels"],
                "tree_pixels": statistics["tree_pixels"],
                "background_pixels": statistics["background_pixels"],
                "tree_percentage": round(
                    statistics["tree_percentage"],
                    4,
                ),
                "background_percentage": round(
                    statistics["background_percentage"],
                    4,
                ),
                "instance_count": statistics["instance_count"],
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
            }

            all_statistics.append(row)

            print("\nHASIL PREDIKSI")
            print(
                f"Persentase pohon : "
                f"{statistics['tree_percentage']:.2f}%"
            )
            print(
                f"Persentase background : "
                f"{statistics['background_percentage']:.2f}%"
            )
            print(
                f"Jumlah instance/pohon : "
                f"{statistics['instance_count']}"
            )
            print(
                f"Rata-rata luas instance : "
                f"{statistics['mean_instance_area_pixel']:.2f} piksel"
            )

            print("\nOutput:")
            print(f"  Semantic          : {semantic_path}")
            print(f"  Instance          : {instance_path}")
            print(f"  Overlay           : {overlay_path}")
            print(
                f"  Semantic preview  : "
                f"{semantic_preview_path}"
            )
            print(
                f"  Instance preview  : "
                f"{instance_preview_path}"
            )
            print(f"  Report            : {report_path}")

            berhasil += 1

            # Bersihkan memori setiap selesai satu citra
            del image
            del semantic_mask
            del instance_mask
            del overlay
            del semantic_preview
            del instance_preview

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            gagal += 1

            print(
                f"GAGAL memproses {image_path.name}"
            )
            print(f"Error: {e}")

    # Simpan rekap seluruh citra
    summary_csv_path = (
        STATISTICS_OUTPUT_DIR
        / "summary_inference.csv"
    )

    save_statistics_csv(
        all_statistics,
        summary_csv_path,
    )

    print()
    print("=" * 75)
    print("BATCH INFERENCE SELESAI")
    print("=" * 75)
    print(f"Berhasil       : {berhasil}")
    print(f"Gagal          : {gagal}")
    print(f"Total citra    : {len(image_paths)}")
    print(f"Rekap CSV      : {summary_csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()