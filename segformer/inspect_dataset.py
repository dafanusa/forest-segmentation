from pathlib import Path

from datasets import load_dataset


# ============================================================
# PATH DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "tcd_dataset"
    / "data"
)


# ============================================================
# LOAD DATASET
# ============================================================

def main():

    print("=" * 70)
    print("INSPEKSI DATASET TCD")
    print("=" * 70)

    print(f"Dataset root: {DATASET_ROOT}")

    train_files = sorted(
        str(p)
        for p in DATASET_ROOT.glob("train-*.parquet")
    )

    test_files = sorted(
        str(p)
        for p in DATASET_ROOT.glob("test-*.parquet")
    )

    if not train_files:
        raise FileNotFoundError(
            "File train-*.parquet tidak ditemukan."
        )

    if not test_files:
        raise FileNotFoundError(
            "File test-*.parquet tidak ditemukan."
        )

    print(f"\nJumlah file train: {len(train_files)}")
    print(f"Jumlah file test : {len(test_files)}")

    print("\nMemuat dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
            "test": test_files,
        },
    )

    print("\n" + "=" * 70)
    print("INFORMASI DATASET")
    print("=" * 70)

    print(dataset)

    print("\nJumlah data:")
    print(f"Train: {len(dataset['train'])}")
    print(f"Test : {len(dataset['test'])}")

    print("\nKolom dataset:")
    print(dataset["train"].column_names)

    print("\nInformasi fitur:")
    print(dataset["train"].features)

    print("\nContoh data:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()