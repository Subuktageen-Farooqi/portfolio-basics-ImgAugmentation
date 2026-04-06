# Image Augmentation for Deep Learning

This repository explores **image augmentation** as a practical way to increase dataset size and variability when collecting large volumes of images is difficult. The core idea, is that a single source image can be transformed in many realistic ways to simulate variation that may occur in real-world data. Instead of relying only on raw images, we can expose a model to altered versions with different geometry, framing, orientation, and appearance. This improves diversity in the dataset and can help models generalize better.
### Why augmentation matters
Data augmentation is useful when:

- collecting more labeled images is expensive or impractical
- the available dataset is too small
- the model needs to be more robust to visual variation
- we want to reduce over-reliance on narrow image patterns


## Overview

In deep learning, model performance often depends heavily on the quantity and diversity of training data. When a dataset is small or lacks variation, augmentation helps by creating modified versions of existing images while preserving their semantic identity thus reducing over-reliance on narrow image patterns.

This repo contains implementations of image augmentation in:

- **Keras / TensorFlow** for single-image augmentation
- **PyTorch / torchvision** for single-image and bulk folder-based augmentation


## Project Goals

- the **concept** of image augmentation
- the **augmentation techniques** used in practice
- the **differnt framework implementations**
- the **resulting synthetic image diversity** produced from a small input set

The project demonstrates how common augmentation operations can be combined into reusable pipelines that generate multiple transformed outputs from one or more source images.


## Files in This Repository

- `image_augmentation_keras.py` : Single-image augmentation using **Keras `ImageDataGenerator`**. It loads one image, converts it into an array, reshapes it into batch format, and generates **40 augmented images** saved to an output folder.

- `image_augmentation_pytorch.py` : Single-image augmentation using **PyTorch / torchvision transforms**. It builds a stochastic augmentation pipeline and saves **40 transformed outputs** from one input image.

- `bulk_image_augmentation_pytorch.py` : Folder-based augmentation using **PyTorch / torchvision**. It scans a folder of valid image files and generates **40 augmented images per source image**, making it suitable for dataset expansion workflows.


## Techniques Used

Across the tutorial and the code files, the following augmentation techniques are used.

### 1. Rotation
Images are randomly rotated to simulate viewpoint changes and orientation variation.

**Insert image here:** original image vs rotated augmented sample


### 2. Shear / Affine Transformation
Shear is a geometric distortion, while the PyTorch scripts use `RandomAffine` with translation, scaling, and shear to produce more flexible spatial perturbations.

**Insert image here:** affine/sheared augmentation examples


### 3. Zoom / Resized Cropping
Basic random zooming through Keras, while the PyTorch implementations use `RandomResizedCrop`, which changes crop region and scale while preserving the target output size. This helps simulate framing differences and partial object emphasis.

**Insert image here:** zoomed / cropped augmentation samples


### 4. Horizontal Flip
Horizontal flipping is used to mirror the image, increasing directional variation in the dataset.

**Insert image here:** original image vs horizontally flipped version


### 5. Brightness Adjustment
Both Keras and PyTorch code apply brightness-related transformations to simulate lighting changes. In PyTorch this is done via `ColorJitter`; in Keras it is done through `brightness_range`.

**Insert image here:** brightness-augmented examples


### 6. Contrast and Saturation Adjustment
The PyTorch implementations use `ColorJitter` to vary **contrast** and **saturation**, extending beyond the more basic Keras parameter set. This broadens appearance diversity and makes the PyTorch pipeline more expressive. 

**Insert image here:** contrast/saturation variation examples


## Framework Implementations

## Keras Approach

1. define the output folder  
2. initialize `ImageDataGenerator` with augmentation parameters  
3. load the image  
4. convert it to a NumPy array  
5. reshape it into batch format  
6. generate and save augmented outputs  

The script saves **40 augmented images** from a single source image in the same directory as the source image.


## PyTorch Approach

The PyTorch implementation builds a transform pipeline using `torchvision.transforms`. Instead of relying on a generator loop like Keras, it repeatedly applies a stochastic transform composition to the source image and saves the results.

The single-image PyTorch script produces **40 augmented outputs**, while the bulk version applies the same idea across all valid images in a folder. The bulk script supports `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files.


## Example Input Images

Two example sunflower images were used to demonstrate augmentation behavior on bright natural imagery with strong petal structure, texture, and color contrast.

**Insert image here:** source image 1  
**Insert image here:** source image 2


## Results

The main result of this project is the successful generation of diverse augmented image sets from limited source data.

### Output Summary

- **Keras script:** generates 40 augmented images from one input image. :contentReference[oaicite:31]{index=31}
- **PyTorch single-image script:** generates 40 augmented images from one input image. :contentReference[oaicite:32]{index=32}
- **PyTorch bulk script:** generates 40 augmented images for each valid image in a folder. :contentReference[oaicite:33]{index=33}


## Observations

Some practical observations from the repo:

- **Keras version** is simpler and closely matches tutorial-style educational workflows.
- **PyTorch version** gives finer control over augmentation design through transform composition. 
- **Bulk augmentation** is more useful when preparing a larger dataset for training, because it scales the pipeline from one image to many.

The PyTorch implementation is also slightly more advanced than the TF-Keras baseline because it includes affine translation, scaling, resized cropping, and color jitter controls.


## Limitations

This project focuses on **augmentation generation**, not on full model training or evaluation. So while it clearly demonstrates the mechanics of augmentation, it does not yet quantify downstream impact on classification accuracy, generalization, or robustness.

A strong next step would be to train the same model:

- once on original data only
- once on augmented data

and compare validation performance.
