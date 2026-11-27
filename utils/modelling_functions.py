import os

import numpy as np
import tensorflow as tf


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

    # Keep pixels in [0, 255]; the model applies Rescaling.
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)