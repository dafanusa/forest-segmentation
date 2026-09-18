from __future__ import annotations

from pathlib import Path
from typing import Any

import io
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from datasets import load_dataset

from .config_deeplabv3plus import (
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
# RANDOM SEED
# ============================================================

def seed_worker(worker_id: int) -> None:
    """
    Mengatur seed untuk setiap worker DataLoader.
    """
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# IMAGE CONVERSION
# ============================================================

def convert_to_pil(value: Any) -> Image.Image:
    """
    Mengubah berbagai format data gambar menjadi PIL Image.

    Format yang didukung:
    - PIL.Image
    - dictionary Hugging Face Image:
      {"bytes": ..., "path": ...}
    - bytes
    - numpy array
    - torch.Tensor
    - path file gambar
    """

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")

        if image_bytes is not None:
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()

            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image_path is not None:
            return Image.open(image_path).convert("RGB")

        raise ValueError(
            "Dictionary gambar tidak memiliki key 'bytes' atau 'path'."
        )

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, np.ndarray):
        array = value

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")

        return Image.fromarray(array).convert("RGB")

    if torch.is_tensor(value):
        tensor = value.detach().cpu()

        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)

        array = tensor.numpy()

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")

    raise TypeError(
        f"Format gambar tidak didukung: {type(value)}"
    )


# ============================================================
# MASK CONVERSION
# ============================================================

def convert_to_mask(value: Any) -> Image.Image:
    """
    Mengubah data mask menjadi PIL grayscale.

    Mask akhir akan memiliki:
    - 0 untuk background
    - 1 untuk tree

    Apabila nilai mask berupa 0 dan 255, akan dikonversi
    menjadi 0 dan 1.
    """

    if isinstance(value, Image.Image):
        mask = value.convert("L")

    elif isinstance(value, dict):
        mask_bytes = value.get("bytes")
        mask_path = value.get("path")

        if mask_bytes is not None:
            if isinstance(mask_bytes, memoryview):
                mask_bytes = mask_bytes.tobytes()

            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        elif mask_path is not None:
            mask = Image.open(mask_path).convert("L")

        else:
            raise ValueError(
                "Dictionary mask tidak memiliki key 'bytes' atau 'path'."
            )

    elif isinstance(value, bytes):
        mask = Image.open(io.BytesIO(value)).convert("L")

    elif isinstance(value, np.ndarray):
        array = value

        if array.ndim == 3:
            if array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))

            if array.shape[-1] > 1:
                array = np.max(array, axis=-1)

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)

        mask = Image.fromarray(array).convert("L")

    elif torch.is_tensor(value):
        tensor = value.detach().cpu()

        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.squeeze(0)

            if tensor.ndim == 3:
                tensor = tensor.permute(1, 2, 0)
                tensor = tensor.max(dim=-1).values

        array = tensor.numpy()

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)

        mask = Image.fromarray(array).convert("L")

    elif isinstance(value, (str, Path)):
        mask = Image.open(value).convert("L")

    else:
        raise TypeError(
            f"Format mask tidak didukung: {type(value)}"
        )

    return mask


# ============================================================
# DATASET COLUMN DETECTION
# ============================================================

def find_column(
    column_names: list[str],
    preferred_name: str,
    candidates: list[str],
) -> str:
    """
    Mencari nama kolom dataset secara fleksibel.
    """

    if preferred_name in column_names:
        return preferred_name

    lower_mapping = {
        column.lower(): column
        for column in column_names
    }

    if preferred_name.lower() in lower_mapping:
        return lower_mapping[preferred_name.lower()]

    for candidate in candidates:
        if candidate in column_names:
            return candidate

        if candidate.lower() in lower_mapping:
            return lower_mapping[candidate.lower()]

    raise KeyError(
        f"Kolom '{preferred_name}' tidak ditemukan.\n"
        f"Kolom yang tersedia: {column_names}\n"
        f"Kandidat yang dicoba: {candidates}"
    )


# ============================================================
# SEGMENTATION DATASET
# ============================================================

class TreeSegmentationDataset(Dataset):
    """
    Dataset PyTorch untuk segmentasi pohon.

    Output:
        {
            "pixel_values": Tensor [3, H, W],
            "labels": Tensor [H, W]
        }
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]

        image = convert_to_pil(item[self.image_column])
        mask = convert_to_mask(item[self.mask_column])

        # Pastikan ukuran gambar dan mask sama
        mask = mask.resize(
            image.size,
            resample=Image.Resampling.NEAREST,
        )

        # ----------------------------------------------------
        # AUGMENTASI TRAINING
        # ----------------------------------------------------

        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )

        # ----------------------------------------------------
        # RESIZE
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
        # IMAGE TO TENSOR
        # ----------------------------------------------------

        image_tensor = self.image_transform(image)

        # ----------------------------------------------------
        # MASK TO TENSOR
        # ----------------------------------------------------

        mask_array = np.array(mask, dtype=np.uint8)

        # Semua nilai > 0 dianggap sebagai kelas tree
        mask_array = (mask_array > 0).astype(np.int64)

        mask_tensor = torch.from_numpy(mask_array).long()

        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
        }


# ============================================================
# LOAD HUGGING FACE PARQUET DATASET
# ============================================================

def load_parquet_dataset(
    pattern: str,
    split_name: str,
):
    """
    Membaca seluruh file Parquet berdasarkan pattern.
    """

    files = sorted(DATA_DIR.glob(Path(pattern).name))

    if not files:
        raise FileNotFoundError(
            f"Tidak ditemukan file Parquet untuk pattern:\n{pattern}\n"
            f"Folder yang diperiksa: {DATA_DIR}"
        )

    file_paths = [str(file) for file in files]

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
    print(f"Nama kolom: {dataset.column_names}")

    return dataset


# ============================================================
# BUILD DATALOADERS
# ============================================================

def build_dataloaders():
    """
    Membuat train_loader dan validation_loader.

    Dataset train akan dibagi menjadi:
    - 80% training
    - 20% validation

    sesuai VAL_SIZE pada config.
    """

    full_dataset = load_parquet_dataset(
        pattern=TRAIN_PATTERN,
        split_name="train",
    )

    column_names = list(full_dataset.column_names)

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
    print("KOLOM DATASET")
    print("=" * 80)
    print(f"Image column : {image_column}")
    print(f"Mask column  : {mask_column}")

    # --------------------------------------------------------
    # SPLIT TRAINING DAN VALIDASI
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
    print(f"Data training   : {len(train_hf)}")
    print(f"Data validation : {len(val_hf)}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader


# ============================================================
# TEST DATA LOADER
# ============================================================

def build_test_loader():
    """
    Membuat DataLoader untuk dataset test.
    """

    test_dataset_hf = load_parquet_dataset(
        pattern=TEST_PATTERN,
        split_name="test",
    )

    column_names = list(test_dataset_hf.column_names)

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
        hf_dataset=test_dataset_hf,
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
    print("PEMERIKSAAN BATCH DATASET")
    print("=" * 80)
    print(f"Pixel values shape : {batch['pixel_values'].shape}")
    print(f"Labels shape       : {batch['labels'].shape}")
    print(f"Pixel values dtype : {batch['pixel_values'].dtype}")
    print(f"Labels dtype       : {batch['labels'].dtype}")
    print(f"Unique mask values: {torch.unique(batch['labels'])}")