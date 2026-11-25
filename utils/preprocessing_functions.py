import os

import numpy as np
import pandas as pd  
import tensorflow as tf

from PIL import Image
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



def images_to_tensor_batches(
    filenames, labels, image_dir, batch_size, training=False
):
    paths = [os.path.join(image_dir, filename) for filename in filenames]
    labels = np.asarray(labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    # Decode each JPEG into an RGB tensor.
    dataset = dataset.map(
        lambda path, label: (
            tf.io.decode_jpeg(tf.io.read_file(path), channels=3),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Scale the pixel values to [0, 1].
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)