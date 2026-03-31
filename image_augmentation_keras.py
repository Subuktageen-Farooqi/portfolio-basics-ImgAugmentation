"""
Single-image data augmentation using Keras ImageDataGenerator.
Required packages: tensorflow, pillow, numpy
Run example: python image_augmentation_keras.py
"""

from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Step 2: Define save folder
OUTPUT_DIR = Path("augmented_images")
NUM_AUGMENTATIONS = 40
FILENAME_PREFIX = "keras_aug"

# Step 3: Initialize ImageDataGenerator with tutorial-style augmentation parameters
AUGMENTOR = ImageDataGenerator(
    rotation_range=35,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode="nearest",
)


def augment_with_keras(image_path: Path) -> int:
    """Generate and save 40 augmented images from one source image."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 4: Load image and convert to array
    pil_image = load_img(image_path)
    image_array = img_to_array(pil_image)

    # Step 5: Reshape image for batch processing (N, H, W, C)
    batch = np.expand_dims(image_array, axis=0)

    # Step 6: Generate and save augmented images
    generated = 0
    data_flow = AUGMENTOR.flow(
        batch,
        batch_size=1,
        save_to_dir=str(OUTPUT_DIR),
        save_prefix=FILENAME_PREFIX,
        save_format="jpg",
    )

    for _ in data_flow:
        generated += 1
        if generated >= NUM_AUGMENTATIONS:
            break

    # Step 7: Print save location
    print(f"Source image path: {image_path.resolve()}")
    print(f"Saved {generated} augmented images to: {OUTPUT_DIR.resolve()}")

    return generated


def main() -> None:
    user_input = input("Enter the path to one image: ").strip()
    if not user_input:
        raise ValueError("No image path provided. Please enter a valid image file path.")

    augment_with_keras(Path(user_input).expanduser())


if __name__ == "__main__":
    main()
