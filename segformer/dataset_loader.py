import ast
import json
from io import BytesIO

import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class TCDParquetDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        processor,
        image_size=512,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    # ========================================================
    # IMAGE
    # ========================================================

    def read_image(self, sample):
        image = sample.get("image")

        if image is None:
            raise ValueError("Kolom image tidak ditemukan.")

        # HuggingFace Image menjadi PIL.Image
        if isinstance(image, Image.Image):
            image = np.array(
                image.convert("RGB")
            )

        # Format dictionary: bytes atau path
        elif isinstance(image, dict):

            if image.get("bytes") is not None:
                image = np.array(
                    Image.open(
                        BytesIO(image["bytes"])
                    ).convert("RGB")
                )

            elif image.get("path") is not None:
                image = np.array(
                    Image.open(
                        image["path"]
                    ).convert("RGB")
                )

            else:
                raise ValueError(
                    "Format dictionary pada kolom image tidak dikenali."
                )

        else:
            image = np.array(image)

            if image.ndim == 2:
                image = np.stack(
                    [image] * 3,
                    axis=-1,
                )

            if image.ndim != 3:
                raise ValueError(
                    f"Dimensi image tidak valid: {image.shape}"
                )

            if image.shape[-1] == 1:
                image = np.repeat(
                    image,
                    3,
                    axis=-1,
                )

            elif image.shape[-1] > 3:
                image = image[:, :, :3]

        return image.astype(np.uint8)

    # ========================================================
    # NORMALISASI ANOTASI
    # ========================================================

    def normalize_annotations(self, coco_raw):
        """
        Mengubah berbagai kemungkinan format coco_annotations
        menjadi list berisi dictionary anotasi.
        """

        if coco_raw is None:
            return []

        # Jika berupa string JSON atau representasi Python
        if isinstance(coco_raw, str):

            coco_raw = coco_raw.strip()

            if coco_raw == "":
                return []

            try:
                coco_raw = json.loads(coco_raw)

            except json.JSONDecodeError:
                try:
                    coco_raw = ast.literal_eval(coco_raw)

                except Exception:
                    print(
                        "Peringatan: coco_annotations tidak dapat "
                        "diparsing sebagai JSON."
                    )
                    return []

        # Jika berupa dictionary
        if isinstance(coco_raw, dict):

            # Format COCO lengkap:
            # {"annotations": [...]}
            if "annotations" in coco_raw:
                coco_raw = coco_raw["annotations"]

            # Format satu anotasi:
            # {"segmentation": [...], ...}
            elif "segmentation" in coco_raw:
                coco_raw = [coco_raw]

            else:
                return []

        if not isinstance(coco_raw, list):
            return []

        # Pastikan hanya dictionary yang diproses
        annotations = []

        for ann in coco_raw:
            if isinstance(ann, dict):
                annotations.append(ann)

        return annotations

    # ========================================================
    # COCO SEGMENTATION → DAFTAR POLYGON
    # ========================================================

    def extract_polygons(self, segmentation):
        """
        Mengubah segmentation COCO menjadi list polygon.
        Format yang didukung:

        1. [[x1,y1,x2,y2,...], [...]]
        2. [x1,y1,x2,y2,...]
        """

        if segmentation is None:
            return []

        if not isinstance(segmentation, list):
            return []

        if len(segmentation) == 0:
            return []

        # Format polygon datar:
        # [x1, y1, x2, y2, ...]
        if all(
            isinstance(value, (int, float))
            for value in segmentation
        ):
            return [segmentation]

        # Format nested:
        # [[x1, y1, ...], [x1, y1, ...]]
        polygons = []

        for polygon in segmentation:

            if not isinstance(polygon, list):
                continue

            if len(polygon) < 6:
                continue

            if not all(
                isinstance(value, (int, float))
                for value in polygon
            ):
                continue

            polygons.append(polygon)

        return polygons

    # ========================================================
    # COCO POLYGON → MASK
    # ========================================================

    def build_masks(self, sample, height, width):

        semantic_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        instance_mask = np.zeros(
            (height, width),
            dtype=np.int32,
        )

        coco_raw = sample.get(
            "coco_annotations",
            None,
        )

        annotations = self.normalize_annotations(
            coco_raw
        )

        # ----------------------------------------------------
        # Jika tersedia anotasi COCO polygon
        # ----------------------------------------------------

        if len(annotations) > 0:

            instance_id = 1
            valid_instance_count = 0

            for ann in annotations:

                segmentation = ann.get(
                    "segmentation",
                    None,
                )

                polygons = self.extract_polygons(
                    segmentation
                )

                if len(polygons) == 0:
                    continue

                current_instance_has_polygon = False

                for polygon in polygons:

                    try:
                        points = np.asarray(
                            polygon,
                            dtype=np.float32,
                        ).reshape(-1, 2)

                    except Exception:
                        continue

                    if points.shape[0] < 3:
                        continue

                    # Hilangkan koordinat non-finite
                    if not np.isfinite(points).all():
                        continue

                    # Bulatkan koordinat polygon
                    points = np.round(
                        points
                    ).astype(np.int32)

                    # Batasi koordinat agar tidak keluar gambar
                    points[:, 0] = np.clip(
                        points[:, 0],
                        0,
                        width - 1,
                    )

                    points[:, 1] = np.clip(
                        points[:, 1],
                        0,
                        height - 1,
                    )

                    # Minimal tiga titik unik untuk polygon
                    if len(np.unique(points, axis=0)) < 3:
                        continue

                    # Semantic segmentation:
                    # background = 0
                    # tree/object = 1
                    cv2.fillPoly(
                        semantic_mask,
                        [points],
                        1,
                    )

                    # Instance segmentation:
                    # setiap anotasi memiliki ID berbeda
                    cv2.fillPoly(
                        instance_mask,
                        [points],
                        instance_id,
                    )

                    current_instance_has_polygon = True

                if current_instance_has_polygon:
                    instance_id += 1
                    valid_instance_count += 1

            # Jika COCO ada tetapi tidak menghasilkan polygon valid,
            # gunakan fallback annotation
            if valid_instance_count > 0:
                return semantic_mask, instance_mask

        # ----------------------------------------------------
        # Fallback: annotation berupa mask
        # ----------------------------------------------------

        annotation = sample.get(
            "annotation",
            None,
        )

        if annotation is None:
            raise ValueError(
                "Dataset tidak memiliki anotasi valid pada "
                "coco_annotations maupun annotation."
            )

        # Annotation berupa PIL Image
        if isinstance(annotation, Image.Image):

            mask = np.array(
                annotation.convert("L")
            )

        # Annotation berupa dictionary
        elif isinstance(annotation, dict):

            if annotation.get("bytes") is not None:

                mask = np.array(
                    Image.open(
                        BytesIO(annotation["bytes"])
                    ).convert("L")
                )

            elif annotation.get("path") is not None:

                mask = np.array(
                    Image.open(
                        annotation["path"]
                    ).convert("L")
                )

            else:
                raise ValueError(
                    "Format dictionary pada kolom annotation "
                    "tidak dikenali."
                )

        else:
            mask = np.array(annotation)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.ndim != 2:
            raise ValueError(
                f"Dimensi annotation mask tidak valid: {mask.shape}"
            )

        # Samakan ukuran annotation dengan image
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        semantic_mask = (
            mask > 0
        ).astype(np.uint8)

        # Instance mask fallback menggunakan connected components
        _, instance_mask = cv2.connectedComponents(
            semantic_mask,
            connectivity=8,
        )

        instance_mask = instance_mask.astype(
            np.int32
        )

        return semantic_mask, instance_mask

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        try:

            sample = self.dataset[idx]

            image = self.read_image(
                sample
            )

            height, width = image.shape[:2]

            semantic_mask, instance_mask = self.build_masks(
                sample,
                height,
                width,
            )

            # Resize image
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

            # Resize semantic mask
            semantic_mask = cv2.resize(
                semantic_mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

            # Resize instance mask
            instance_mask = cv2.resize(
                instance_mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

            # Pastikan tipe data sesuai
            semantic_mask = semantic_mask.astype(
                np.int64
            )

            instance_mask = instance_mask.astype(
                np.int64
            )

            encoded = self.processor(
                images=image,
                segmentation_maps=semantic_mask,
                return_tensors="pt",
            )

            pixel_values = encoded[
                "pixel_values"
            ].squeeze(0)

            labels = encoded[
                "labels"
            ].squeeze(0)

            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "semantic_mask": torch.from_numpy(
                    semantic_mask
                ).long(),
                "instance_mask": torch.from_numpy(
                    instance_mask
                ).long(),
                "image_path": str(idx),
            }

        except Exception as e:

            print("\n" + "=" * 80)
            print("ERROR SAAT MEMPROSES DATASET")
            print("Index data:", idx)
            print("Error:", repr(e))
            print("=" * 80)

            raise