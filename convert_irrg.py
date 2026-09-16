from pathlib import Path
from PIL import Image

input_dir = Path("dataset/data/img_irrg")

for png_file in input_dir.glob("*.png"):
    jpg_file = png_file.with_suffix(".jpg")

    image = Image.open(png_file).convert("RGB")
    image.save(jpg_file, "JPEG", quality=95)

    print(f"Converted: {png_file.name} -> {jpg_file.name}")