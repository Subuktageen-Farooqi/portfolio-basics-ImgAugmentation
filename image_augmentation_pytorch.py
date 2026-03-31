"""
Single-image data augmentation using PyTorch/torchvision.
Required packages: torch, torchvision, pillow
Run example: python image_augmentation_pytorch.py
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Configuration
OUTPUT_DIR = Path("augmented_images")
NUM_AUGMENTATIONS = 40
OUTPUT_PREFIX = "single_aug"
JPEG_QUALITY = 95

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


def build_augmentation_pipeline(target_size: tuple[int, int]) -> transforms.Compose:
    """Create a stochastic augmentation pipeline for one image."""
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


def augment_single_image(image_path: Path) -> int:
    """Generate and save augmented versions of one input image."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as source:
            original_image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unable to open image '{image_path}': {exc}") from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = build_augmentation_pipeline(original_image.size[::-1])

    for index in range(1, NUM_AUGMENTATIONS + 1):
        augmented = pipeline(original_image)
        output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{index}.jpg"
        augmented.save(output_path, format="JPEG", quality=JPEG_QUALITY)

    print(f"Source image path: {image_path.resolve()}")
    print(f"Output folder path: {OUTPUT_DIR.resolve()}")
    print(f"Total saved count: {NUM_AUGMENTATIONS}")

    return NUM_AUGMENTATIONS


def main() -> None:
    user_input = input("Enter the path to one image: ").strip()
    if not user_input:
        raise ValueError("No image path provided. Please enter a valid image file path.")

    augment_single_image(Path(user_input).expanduser())


if __name__ == "__main__":
    main()
