"""Modul Transform: preprocessing dataset Estimation of Obesity Levels."""

import tensorflow as tf
import tensorflow_transform as tft

from features import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)


def _fill_in_missing(tensor):
    """Mengubah SparseTensor menjadi Tensor padat berukuran [batch, 1]."""
    if isinstance(tensor, tf.sparse.SparseTensor):
        default_value = "" if tensor.dtype == tf.string else 0
        tensor = tf.sparse.to_dense(
            tf.SparseTensor(
                tensor.indices, tensor.values, [tensor.dense_shape[0], 1]
            ),
            default_value,
        )
    return tensor


def preprocessing_fn(inputs):
    """Fungsi preprocessing TFT.

    Args:
        inputs: map nama fitur mentah ke Tensor/SparseTensor.

    Returns:
        Map nama fitur hasil transformasi ke Tensor.
    """
    outputs = {}

    # Fitur numerik: standardisasi z-score.
    for key in NUMERICAL_FEATURES:
        values = tf.cast(_fill_in_missing(inputs[key]), tf.float32)
        outputs[transformed_name(key)] = tf.reshape(
            tft.scale_to_z_score(values), [-1]
        )

    # Fitur kategorikal: vocabulary lalu one-hot (kategori + 1 bucket OOV).
    for key, cardinality in CATEGORICAL_FEATURES.items():
        values = _fill_in_missing(inputs[key])
        ids = tft.compute_and_apply_vocabulary(
            tf.reshape(values, [-1]), num_oov_buckets=1, vocab_filename=key
        )
        outputs[transformed_name(key)] = tf.one_hot(
            ids, depth=cardinality + 1, dtype=tf.float32
        )

    # Label: string menjadi indeks integer.
    label = tf.reshape(_fill_in_missing(inputs[LABEL_KEY]), [-1])
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        label, vocab_filename="label_vocab"
    )

    return outputs
