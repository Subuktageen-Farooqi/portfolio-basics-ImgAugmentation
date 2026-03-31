"""
Bulk image data augmentation using PyTorch/torchvision.
Required packages: torch, torchvision, pillow
Run example: python bulk_image_augmentation_pytorch.py
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Configuration
OUTPUT_DIR = Path("augmented_images")
NUM_AUGMENTATIONS_PER_IMAGE = 40
JPEG_QUALITY = 95
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Augmentation settings (tutorial-style)
ROTATION_DEGREES = 35
AFFINE_DEGREES = 12
AFFINE_TRANSLATE = (0.08, 0.08)
AFFINE_SCALE = (0.85, 1.15)
AFFINE_SHEAR = 15
RESIZED_CROP_SCALE = (0.8, 1.0)
RESIZED_CROP_RATIO = (0.9, 1.1)
H_FLIP_PROBABILITY = 0.5
BRIGHTNESS = 0.3
CONTRAST = 0.2
SATURATION = 0.2


def is_valid_image_file(path: Path) -> bool:
    """Check extension first to filter likely image files."""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def build_augmentation_pipeline(target_size: tuple[int, int]) -> transforms.Compose:
    """Create a stochastic augmentation pipeline for one image size."""
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.RandomAffine(
                degrees=AFFINE_DEGREES,
                translate=AFFINE_TRANSLATE,
                scale=AFFINE_SCALE,
                shear=AFFINE_SHEAR,
            ),
            transforms.RandomResizedCrop(
                size=target_size,
                scale=RESIZED_CROP_SCALE,
                ratio=RESIZED_CROP_RATIO,
            ),
            transforms.RandomHorizontalFlip(p=H_FLIP_PROBABILITY),
            transforms.ColorJitter(
                brightness=BRIGHTNESS,
                contrast=CONTRAST,
                saturation=SATURATION,
            ),
        ]
    )


def collect_valid_images(folder_path: Path) -> list[Path]:
    """Return sorted image paths, ignoring non-image files safely."""
    files = sorted(folder_path.iterdir(), key=lambda p: p.name.lower())
    return [path for path in files if is_valid_image_file(path)]


def augment_folder_images(input_folder: Path) -> int:
    """Generate 40 augmented images per source image."""
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    source_images = collect_valid_images(input_folder)
    if not source_images:
        raise ValueError(
            f"No valid image files found in folder: {input_folder}\n"
            f"Supported extensions: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for source_index, image_path in enumerate(source_images, start=1):
        try:
            with Image.open(image_path) as source:
                original_image = source.convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Skip unreadable files safely even if extension looks valid.
            continue

        pipeline = build_augmentation_pipeline(original_image.size[::-1])

        for aug_index in range(1, NUM_AUGMENTATIONS_PER_IMAGE + 1):
            augmented = pipeline(original_image)
            output_name = f"image_{source_index}_{aug_index}.jpg"
            output_path = OUTPUT_DIR / output_name
            augmented.save(output_path, format="JPEG", quality=JPEG_QUALITY)
            total_saved += 1

    if total_saved == 0:
        raise ValueError("No augmented images were generated. Check that images are readable.")

    print(f"Input folder path: {input_folder.resolve()}")
    print(f"Number of valid source images found: {len(source_images)}")
    print(f"Total augmented images generated: {total_saved}")
    print(f"Output folder path: {OUTPUT_DIR.resolve()}")

    return total_saved


def main() -> None:
    user_input = input("Enter the path to the folder containing images: ").strip()
    if not user_input:
        raise ValueError("No folder path provided. Please enter a valid folder path.")

    augment_folder_images(Path(user_input).expanduser())


if __name__ == "__main__":
    main()
