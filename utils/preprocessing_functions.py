import numpy as np
import pandas as pd  


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