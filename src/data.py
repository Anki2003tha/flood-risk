"""Data utilities for the Flood Risk Prediction project.

This module provides a small synthetic dataset generator (tf.data) useful for
development and unit tests. Replace or extend the `get_dataset` function with
actual Sentinel loaders (rasterio/xarray) when real data is available.
"""

from typing import Tuple


def get_synthetic_tf_dataset(num_samples: int = 32, img_size=(128, 128, 3), num_classes: int = 3, batch_size: int = 8):
    """Return (train_ds, val_ds) as tf.data.Dataset objects built from random arrays.

    The function imports TensorFlow lazily so importing this module doesn't
    require TensorFlow to be installed unless the function is called.
    """
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow (and numpy) are required to create the synthetic dataset: " + str(e))

    # create arrays
    imgs = np.random.rand(num_samples, img_size[0], img_size[1], img_size[2]).astype("float32")
    labels = np.random.randint(0, num_classes, size=(num_samples,)).astype("int32")

    # split
    split = int(num_samples * 0.8)
    x_train, x_val = imgs[:split], imgs[split:]
    y_train, y_val = labels[:split], labels[split:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(64).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def get_dataset(mode: str = "synthetic", **kwargs) -> Tuple[object, object]:
    """Main entry for dataset creation.

    mode: 'synthetic' (default) or 'real' (not implemented)
    Returns: (train_ds, val_ds)
    """
    if mode == "synthetic":
        return get_synthetic_tf_dataset(**kwargs)
    else:
        raise NotImplementedError("Real dataset loader not yet implemented. Use 'synthetic' for demo.")
