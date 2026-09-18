from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from datasets import load_dataset

from .config_upernet import (
    DATA_DIR,
    TRAIN_PATTERN,
    TEST_PATTERN,
    IMAGE_SIZE,
    IMAGE_COLUMN,
    MASK_COLUMN,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SIZE,
    SEED,
)


# ============================================================
# SEED WORKER
# ============================================================

def seed_worker(worker_id: int) -> None:
    """
    Mengatur seed untuk setiap worker DataLoader.
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CONVERT IMAGE TO PIL
# ============================================================

def convert_to_pil(value: Any) -> Image.Image:
    """
    Mengubah data image dari Parquet/Hugging Face menjadi PIL RGB.
    """

    # Jika sudah berupa PIL Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    # Jika berupa dictionary dari Hugging Face Image feature
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")

        # Data image berupa bytes
        if image_bytes is not None:
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()

            return Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        # Data image berupa path
        if image_path is not None:
            return Image.open(image_path).convert("RGB")

        raise ValueError(
            "Data image berbentuk dictionary, tetapi tidak memiliki "
            "key 'bytes' atau 'path'."
        )

    # Jika berupa bytes
    if isinstance(value, bytes):
        return Image.open(
            io.BytesIO(value)
        ).convert("RGB")

    # Jika berupa NumPy array
    if isinstance(value, np.ndarray):
        array = value

        # Jika format CHW, ubah menjadi HWC
        if array.ndim == 3:
            if array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))

        # Konversi ke uint8
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    # Jika berupa Tensor
    if torch.is_tensor(value):
        tensor = value.detach().cpu()

        # Jika format CHW, ubah menjadi HWC
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)

        array = tensor.numpy()

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    # Jika berupa path
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")

    raise TypeError(
        f"Tipe image tidak didukung: {type(value)}"
    )


# ============================================================
# CONVERT MASK TO PIL
# ============================================================

def convert_to_mask(value: Any) -> Image.Image:
    """
    Mengubah data mask menjadi PIL grayscale.

    Nilai mask akhir:
        0 = background
        1 = tree
    """

    # Jika sudah berupa PIL Image
    if isinstance(value, Image.Image):
        return value.convert("L")

    # Jika berupa dictionary dari Hugging Face
    if isinstance(value, dict):
        mask_bytes = value.get("bytes")
        mask_path = value.get("path")

        # Data mask berupa bytes
        if mask_bytes is not None:
            if isinstance(mask_bytes, memoryview):
                mask_bytes = mask_bytes.tobytes()

            return Image.open(
                io.BytesIO(mask_bytes)
            ).convert("L")

        # Data mask berupa path
        if mask_path is not None:
            return Image.open(mask_path).convert("L")

        raise ValueError(
            "Data mask tidak memiliki key 'bytes' atau 'path'."
        )

    # Jika berupa bytes
    if isinstance(value, bytes):
        return Image.open(
            io.BytesIO(value)
        ).convert("L")

    # Jika berupa NumPy array
    if isinstance(value, np.ndarray):
        array = value

        # Jika mask memiliki channel tambahan
        if array.ndim == 3:
            if array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))

            if array.ndim == 3:
                array = np.max(array, axis=-1)

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        return Image.fromarray(array).convert("L")

    # Jika berupa Tensor
    if torch.is_tensor(value):
        tensor = value.detach().cpu()

        if tensor.ndim == 3:
            # Jika bentuknya [1, H, W]
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.squeeze(0)

            # Jika masih memiliki channel
            if tensor.ndim == 3:
                tensor = tensor.permute(1, 2, 0)
                tensor = tensor.max(dim=-1).values

        array = tensor.numpy()

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        return Image.fromarray(array).convert("L")

    # Jika berupa path
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("L")

    raise TypeError(
        f"Tipe mask tidak didukung: {type(value)}"
    )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    column_names: list[str],
    preferred_name: str,
    candidates: list[str],
) -> str:
    """
    Mencari nama kolom image atau mask secara fleksibel.
    """

    # Prioritas pertama: nama kolom dari config
    if preferred_name in column_names:
        return preferred_name

    # Pencarian tanpa memperhatikan huruf besar/kecil
    lower_mapping = {
        column.lower(): column
        for column in column_names
    }

    if preferred_name.lower() in lower_mapping:
        return lower_mapping[preferred_name.lower()]

    # Pencarian berdasarkan kandidat
    for candidate in candidates:
        if candidate in column_names:
            return candidate

        if candidate.lower() in lower_mapping:
            return lower_mapping[candidate.lower()]

    raise KeyError(
        f"Kolom '{preferred_name}' tidak ditemukan.\n"
        f"Kolom yang tersedia: {column_names}"
    )


# ============================================================
# DATASET CLASS
# ============================================================

class TreeSegmentationDataset(Dataset):
    """
    Dataset segmentasi pohon untuk UPerNet.

    Output:
        pixel_values: [3, H, W]
        labels      : [H, W]
    """

    def __init__(
        self,
        hf_dataset,
        image_column: str,
        mask_column: str,
        image_size: int = 512,
        training: bool = False,
    ):
        self.dataset = hf_dataset
        self.image_column = image_column
        self.mask_column = mask_column
        self.image_size = image_size
        self.training = training

        # Normalisasi ImageNet untuk backbone Swin Transformer
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        item = self.dataset[index]

        image = convert_to_pil(
            item[self.image_column]
        )

        mask = convert_to_mask(
            item[self.mask_column]
        )

        # Menyamakan ukuran mask dengan image
        mask = mask.resize(
            image.size,
            resample=Image.Resampling.NEAREST,
        )

        # ----------------------------------------------------
        # AUGMENTASI TRAINING
        # ----------------------------------------------------

        if self.training:

            # Horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotasi 90, 180, atau 270 derajat
            if random.random() < 0.5:
                angle = random.choice(
                    [90, 180, 270]
                )

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=(
                        transforms.InterpolationMode.BILINEAR
                    ),
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=(
                        transforms.InterpolationMode.NEAREST
                    ),
                )

        # ----------------------------------------------------
        # RESIZE IMAGE DAN MASK
        # ----------------------------------------------------

        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST,
        )

        # ----------------------------------------------------
        # IMAGE TENSOR
        # ----------------------------------------------------

        image_tensor = self.image_transform(image)

        # ----------------------------------------------------
        # MASK TENSOR
        # ----------------------------------------------------

        mask_array = np.array(
            mask,
            dtype=np.uint8,
        )

        # Semua piksel > 0 menjadi kelas tree
        # 0 = background
        # 1 = tree
        mask_array = (
            mask_array > 0
        ).astype(np.int64)

        mask_tensor = torch.from_numpy(
            mask_array
        ).long()

        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
        }


# ============================================================
# LOAD PARQUET
# ============================================================

def load_parquet_dataset(
    pattern: str,
    split_name: str,
):
    """
    Membaca semua file Parquet berdasarkan pattern.
    """

    pattern_path = Path(pattern)

    files = sorted(
        DATA_DIR.glob(pattern_path.name)
    )

    if not files:
        raise FileNotFoundError(
            f"Tidak ditemukan file Parquet.\n"
            f"Pattern: {pattern}\n"
            f"Folder: {DATA_DIR}"
        )

    file_paths = [
        str(file)
        for file in files
    ]

    print()
    print("=" * 80)
    print(f"MEMUAT DATASET {split_name.upper()}")
    print("=" * 80)

    for file_path in file_paths:
        print(f"- {file_path}")

    dataset = load_dataset(
        "parquet",
        data_files={
            split_name: file_paths,
        },
        split=split_name,
    )

    print(f"Jumlah data {split_name}: {len(dataset)}")
    print(f"Kolom dataset: {dataset.column_names}")

    return dataset


# ============================================================
# BUILD TRAIN DAN VALIDATION LOADER
# ============================================================

def build_dataloaders():
    """
    Membuat train_loader dan val_loader.

    Catatan:
    - train_loader menggunakan drop_last=True agar batch terakhir
      tidak hanya berisi satu data.
    - val_loader menggunakan drop_last=False agar seluruh data
      validasi tetap digunakan.
    """

    full_dataset = load_parquet_dataset(
        pattern=TRAIN_PATTERN,
        split_name="train",
    )

    column_names = list(
        full_dataset.column_names
    )

    image_column = find_column(
        column_names=column_names,
        preferred_name=IMAGE_COLUMN,
        candidates=[
            "image",
            "images",
            "img",
            "rgb",
            "input",
        ],
    )

    mask_column = find_column(
        column_names=column_names,
        preferred_name=MASK_COLUMN,
        candidates=[
            "mask",
            "masks",
            "label",
            "labels",
            "segmentation",
            "annotation",
            "target",
        ],
    )

    print()
    print("=" * 80)
    print("KOLOM DATASET YANG DIGUNAKAN")
    print("=" * 80)
    print(f"Image column: {image_column}")
    print(f"Mask column : {mask_column}")

    # --------------------------------------------------------
    # SPLIT TRAIN DAN VALIDATION
    # --------------------------------------------------------

    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE,
        seed=SEED,
    )

    train_hf = split_dataset["train"]
    val_hf = split_dataset["test"]

    print()
    print("=" * 80)
    print("PEMBAGIAN DATASET")
    print("=" * 80)
    print(f"Training   : {len(train_hf)}")
    print(f"Validation : {len(val_hf)}")

    train_dataset = TreeSegmentationDataset(
        hf_dataset=train_hf,
        image_column=image_column,
        mask_column=mask_column,
        image_size=IMAGE_SIZE,
        training=True,
    )

    val_dataset = TreeSegmentationDataset(
        hf_dataset=val_hf,
        image_column=image_column,
        mask_column=mask_column,
        image_size=IMAGE_SIZE,
        training=False,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    # --------------------------------------------------------
    # TRAINING DATALOADER
    # --------------------------------------------------------
    # Batch size tetap mengikuti config, yaitu 2.
    # drop_last=True mencegah batch terakhir berisi satu data,
    # sehingga BatchNorm pada UPerNet tidak mengalami error.

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # --------------------------------------------------------
    # VALIDATION DATALOADER
    # --------------------------------------------------------
    # Semua data validasi tetap digunakan.

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader


# ============================================================
# BUILD TEST LOADER
# ============================================================

def build_test_loader():
    """
    Membuat DataLoader untuk dataset testing.
    """

    test_hf = load_parquet_dataset(
        pattern=TEST_PATTERN,
        split_name="test",
    )

    column_names = list(
        test_hf.column_names
    )

    image_column = find_column(
        column_names=column_names,
        preferred_name=IMAGE_COLUMN,
        candidates=[
            "image",
            "images",
            "img",
            "rgb",
            "input",
        ],
    )

    mask_column = find_column(
        column_names=column_names,
        preferred_name=MASK_COLUMN,
        candidates=[
            "mask",
            "masks",
            "label",
            "labels",
            "segmentation",
            "annotation",
            "target",
        ],
    )

    test_dataset = TreeSegmentationDataset(
        hf_dataset=test_hf,
        image_column=image_column,
        mask_column=mask_column,
        image_size=IMAGE_SIZE,
        training=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return test_loader


# ============================================================
# DEBUG DATASET
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader = build_dataloaders()

    batch = next(iter(train_loader))

    print()
    print("=" * 80)
    print("PEMERIKSAAN BATCH UPERNET")
    print("=" * 80)

    print(
        f"Pixel values shape: "
        f"{batch['pixel_values'].shape}"
    )

    print(
        f"Labels shape      : "
        f"{batch['labels'].shape}"
    )

    print(
        f"Pixel values dtype: "
        f"{batch['pixel_values'].dtype}"
    )

    print(
        f"Labels dtype      : "
        f"{batch['labels'].dtype}"
    )

    print(
        f"Unique mask values: "
        f"{torch.unique(batch['labels'])}"
    )

    print()
    print("=" * 80)
    print("INFORMASI DATALOADER")
    print("=" * 80)
    print(f"Batch size config : {BATCH_SIZE}")
    print(f"Jumlah batch train: {len(train_loader)}")
    print(f"Jumlah batch val  : {len(val_loader)}")
    print("Train drop_last   : True")
    print("Val drop_last     : False")