from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50

from .config_deeplabv3plus import (
    MODEL_NAME,
    NUM_CLASSES,
    ID2LABEL,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    PATIENCE,
    MIN_DELTA,
)

from .dataset_loader import build_dataloaders


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL
# ============================================================

def build_model() -> nn.Module:
    """
    Membuat model DeepLabV3+ berbasis ResNet50.

    Catatan:
    torchvision menyediakan DeepLabV3, bukan implementasi
    DeepLabV3+ murni. Struktur ini digunakan sebagai model
    segmentasi DeepLab berbasis ResNet50.
    """

    if MODEL_NAME.lower() != "resnet50":
        raise ValueError(
            f"MODEL_NAME saat ini adalah '{MODEL_NAME}'. "
            "Kode ini menggunakan backbone resnet50."
        )

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
    )

    return model


# ============================================================
# LOSS FUNCTION
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss untuk segmentasi biner/multikelas.
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.softmax(logits, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(
            targets,
            num_classes=self.num_classes,
        )

        targets_one_hot = targets_one_hot.permute(
            0,
            3,
            1,
            2,
        ).float()

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities + targets_one_hot
        ).sum(dim=(0, 2, 3))

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Gabungan Cross Entropy dan Dice Loss.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
        )

        return total_loss


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """
    Menghitung pixel accuracy, mean IoU, dan mean Dice.
    """

    predictions = torch.argmax(logits, dim=1)

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    valid_mask = (
        (targets >= 0)
        & (targets < num_classes)
    )

    target_valid = targets[valid_mask]
    prediction_valid = predictions[valid_mask]

    encoded = (
        num_classes * target_valid
        + prediction_valid
    )

    counts = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )

    confusion_matrix = counts.reshape(
        num_classes,
        num_classes,
    )

    true_positive = np.diag(confusion_matrix)

    actual_positive = confusion_matrix.sum(axis=1)
    predicted_positive = confusion_matrix.sum(axis=0)

    union = (
        actual_positive
        + predicted_positive
        - true_positive
    )

    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=union != 0,
    )

    dice_denominator = (
        actual_positive
        + predicted_positive
    )

    dice = np.divide(
        2.0 * true_positive,
        dice_denominator,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=dice_denominator != 0,
    )

    pixel_accuracy = (
        true_positive.sum()
        / max(confusion_matrix.sum(), 1)
    )

    mean_iou = float(np.mean(iou))
    mean_dice = float(np.mean(dice))

    tree_iou = (
        float(iou[1])
        if num_classes > 1
        else float(iou[0])
    )

    tree_dice = (
        float(dice[1])
        if num_classes > 1
        else float(dice[0])
    )

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "tree_iou": tree_iou,
        "tree_dice": tree_dice,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler,
    epoch: int,
) -> dict[str, float]:

    model.train()

    running_loss = 0.0
    metric_values = {
        "pixel_accuracy": [],
        "mean_iou": [],
        "mean_dice": [],
        "tree_iou": [],
        "tree_dice": [],
    }

    progress_bar = tqdm(
        loader,
        desc=f"Training Epoch {epoch}",
        leave=True,
    )

    for batch in progress_bar:
        images = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True,
        )

        masks = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):
            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs["out"]
            else:
                logits = outputs

            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        metrics = calculate_metrics(
            logits=logits,
            targets=masks,
            num_classes=NUM_CLASSES,
        )

        for key in metric_values:
            metric_values[key].append(metrics[key])

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            IoU=f"{metrics['mean_iou']:.4f}",
        )

    epoch_loss = running_loss / max(len(loader), 1)

    epoch_metrics = {
        key: float(np.mean(values))
        for key, values in metric_values.items()
    }

    epoch_metrics["loss"] = epoch_loss

    return epoch_metrics


# ============================================================
# VALIDATION ONE EPOCH
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    epoch: int,
) -> dict[str, float]:

    model.eval()

    running_loss = 0.0
    metric_values = {
        "pixel_accuracy": [],
        "mean_iou": [],
        "mean_dice": [],
        "tree_iou": [],
        "tree_dice": [],
    }

    progress_bar = tqdm(
        loader,
        desc=f"Validation Epoch {epoch}",
        leave=True,
    )

    for batch in progress_bar:
        images = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True,
        )

        masks = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):
            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs["out"]
            else:
                logits = outputs

            loss = criterion(logits, masks)

        running_loss += loss.item()

        metrics = calculate_metrics(
            logits=logits,
            targets=masks,
            num_classes=NUM_CLASSES,
        )

        for key in metric_values:
            metric_values[key].append(metrics[key])

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            IoU=f"{metrics['mean_iou']:.4f}",
        )

    epoch_loss = running_loss / max(len(loader), 1)

    epoch_metrics = {
        key: float(np.mean(values))
        for key, values in metric_values.items()
    }

    epoch_metrics["loss"] = epoch_loss

    return epoch_metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
) -> None:

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "id2label": ID2LABEL,
        "num_classes": NUM_CLASSES,
    }

    torch.save(checkpoint, path)


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> None:

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print("TRAINING DEEPLABV3+ / DEEPLAB RESNET50")
    print("=" * 90)
    print(f"Model       : {MODEL_NAME}")
    print(f"Classes     : {NUM_CLASSES}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Device      : {DEVICE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print("=" * 90)

    # --------------------------------------------------------
    # DATA LOADER
    # --------------------------------------------------------

    train_loader, val_loader = build_dataloaders()

    print()
    print("=" * 90)
    print("DATALOADER BERHASIL DIBUAT")
    print("=" * 90)
    print(f"Jumlah batch training   : {len(train_loader)}")
    print(f"Jumlah batch validation : {len(val_loader)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()
    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        ce_weight=1.0,
        dice_weight=1.0,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = []

    best_val_iou = -float("inf")
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print()
        print("#" * 90)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("#" * 90)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            epoch=epoch,
        )

        scheduler.step(val_metrics["mean_iou"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(epoch_result)

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print(f"HASIL EPOCH {epoch}")
        print("=" * 90)

        print(
            f"Train Loss       : {train_metrics['loss']:.6f}"
        )
        print(
            f"Train Pixel Acc  : {train_metrics['pixel_accuracy']:.6f}"
        )
        print(
            f"Train Mean IoU   : {train_metrics['mean_iou']:.6f}"
        )
        print(
            f"Train Mean Dice  : {train_metrics['mean_dice']:.6f}"
        )
        print(
            f"Train Tree IoU   : {train_metrics['tree_iou']:.6f}"
        )
        print(
            f"Train Tree Dice  : {train_metrics['tree_dice']:.6f}"
        )

        print("-" * 90)

        print(
            f"Val Loss         : {val_metrics['loss']:.6f}"
        )
        print(
            f"Val Pixel Acc    : {val_metrics['pixel_accuracy']:.6f}"
        )
        print(
            f"Val Mean IoU     : {val_metrics['mean_iou']:.6f}"
        )
        print(
            f"Val Mean Dice    : {val_metrics['mean_dice']:.6f}"
        )
        print(
            f"Val Tree IoU     : {val_metrics['tree_iou']:.6f}"
        )
        print(
            f"Val Tree Dice    : {val_metrics['tree_dice']:.6f}"
        )

        print("-" * 90)
        print(f"Learning Rate    : {current_lr:.8f}")

        # ----------------------------------------------------
        # SAVE LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            val_metrics["mean_iou"]
            > best_val_iou + MIN_DELTA
        ):
            best_val_iou = val_metrics["mean_iou"]
            epochs_without_improvement = 0

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            print()
            print(
                f"CHECKPOINT TERBAIK DISIMPAN: "
                f"{BEST_MODEL_PATH}"
            )
            print(
                f"Best Validation Mean IoU: "
                f"{best_val_iou:.6f}"
            )

        else:
            epochs_without_improvement += 1

            print()
            print(
                "Tidak ada peningkatan Validation Mean IoU."
            )
            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history_path = LOG_DIR / "training_history.json"

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # EARLY STOPPING CHECK
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:
            print()
            print("=" * 90)
            print("EARLY STOPPING AKTIF")
            print("=" * 90)
            print(
                f"Training dihentikan pada epoch {epoch}."
            )
            print(
                f"Validation Mean IoU tidak meningkat "
                f"selama {PATIENCE} epoch."
            )
            break

    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("TRAINING SELESAI")
    print("=" * 90)
    print(f"Best model  : {BEST_MODEL_PATH}")
    print(f"Last model  : {LAST_MODEL_PATH}")
    print(f"History     : {LOG_DIR / 'training_history.json'}")
    print(f"Device      : {DEVICE}")
    print("=" * 90)


if __name__ == "__main__":
    main()