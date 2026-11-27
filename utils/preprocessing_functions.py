import os

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageFilter
from scipy.fft import dctn
from tqdm.auto import tqdm


def compute_perceptual_hash(image):
    grey = image.convert("L").resize(
        (32, 32),
        Image.Resampling.LANCZOS
    )

    pixels = np.asarray(grey, dtype=np.float32)

    coefficients = dctn(
        pixels,
        norm="ortho"
    )[:8, :8]

    median = np.median(coefficients.ravel()[1:])
    bits = coefficients >= median

    return int.from_bytes(
        np.packbits(bits.ravel()).tobytes(),
        byteorder="big"
    )


def restore_image(image, denoise=True, sharpen=True, median_size=3,
                  sharpen_radius=1.0, sharpen_percent=100, sharpen_threshold=3):
    """Class 16 filters: median denoising followed by unsharp masking.

    Accept a PIL image and return a new RGB image at the original size.
    Sharpening enhances edges; it does not reconstruct lost image detail.
    Disable both filters for an unprocessed baseline.
    """
    if median_size < 3 or median_size % 2 == 0:
        raise ValueError("median_size must be an odd integer >= 3")
    if sharpen_radius <= 0 or sharpen_percent < 0 or sharpen_threshold < 0:
        raise ValueError("Invalid unsharp mask parameters")
    result = image.convert("RGB")
    if denoise:
        result = result.filter(ImageFilter.MedianFilter(size=median_size))
    if sharpen:
        result = result.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius, percent=sharpen_percent,
            threshold=sharpen_threshold,
        ))
    return result


def images_to_tensor_batches(filenames, targets, directory, batch_size=64,
                             training=False, quality_options=None,
                             quality_manifest=None, quality_mode="baseline"):
    """Load RGB images lazily, with baseline or per-image selective filtering."""
    

    if quality_mode not in {"baseline", "selective"}:
        raise ValueError("quality_mode must be 'baseline' or 'selective'")
    if quality_mode == "selective" and quality_manifest is None:
        raise ValueError("quality_manifest is required in selective mode")

    filenames = list(filenames)
    targets = np.asarray(targets)
    if len(filenames) != len(targets):
        raise ValueError("filenames and targets must have the same length")
    paths = [os.path.join(directory, name) for name in filenames]
    options = {} if quality_options is None else dict(quality_options)

    if quality_mode == "selective":
        required = {"filename", "apply_denoise", "apply_sharpen"}
        if not required.issubset(quality_manifest.columns):
            raise ValueError("quality_manifest is missing required columns")
        if quality_manifest["filename"].duplicated().any():
            raise ValueError("quality_manifest filenames must be unique")
        actions = quality_manifest.set_index("filename")[[
            "apply_denoise", "apply_sharpen"
        ]].to_dict("index")
        missing = set(filenames) - set(actions)
        if missing:
            raise ValueError(f"quality_manifest is missing {len(missing)} filenames")
    else:
        actions = {}

    def load_image(filepath):
        image_path = os.fsdecode(filepath.item())
        filename = os.path.basename(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if quality_mode == "selective":
                flags = actions[filename]
                if flags["apply_denoise"] or flags["apply_sharpen"]:
                    image = restore_image(
                        image,
                        denoise=bool(flags["apply_denoise"]),
                        sharpen=bool(flags["apply_sharpen"]),
                        **options,
                    )
            return np.asarray(image, dtype=np.float32) / 255.0

    def load_pair(filepath, target):
        pixels = tf.numpy_function(load_image, [filepath], tf.float32)
        pixels.set_shape([None, None, 3])
        return pixels, target

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training and filenames:
        dataset = dataset.shuffle(len(filenames), seed=42)
    return (dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))
