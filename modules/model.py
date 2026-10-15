"""Pembuatan model, pemuatan data, dan penyimpanan model.

Dipakai bersama oleh modul Tuner dan Trainer agar tidak ada kode duplikat.
"""

import tensorflow as tf
from tfx_bsl.public import tfxio

from features import (
    BATCH_SIZE,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUM_CLASSES,
    NUMERICAL_FEATURES,
    transformed_name,
)


def input_fn(file_pattern, data_accessor, tf_transform_output,
             batch_size=BATCH_SIZE):
    """Membuat tf.data.Dataset dari contoh hasil Transform."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=transformed_name(LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    )


def build_model(hparams):
    """Membangun dan meng-compile model klasifikasi.

    Args:
        hparams: objek keras_tuner.HyperParameters. Nilai yang belum
            ditentukan akan memakai nilai default masing-masing.

    Returns:
        tf.keras.Model yang sudah di-compile.
    """
    inputs = {}
    for key in NUMERICAL_FEATURES:
        name = transformed_name(key)
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
    for key, cardinality in CATEGORICAL_FEATURES.items():
        name = transformed_name(key)
        inputs[name] = tf.keras.Input(
            shape=(cardinality + 1,), name=name, dtype=tf.float32
        )

    units_1 = hparams.Int("units_1", min_value=64, max_value=256, step=64)
    units_2 = hparams.Int("units_2", min_value=32, max_value=128, step=32)
    dropout = hparams.Float("dropout", min_value=0.1, max_value=0.4, step=0.1)
    learning_rate = hparams.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    layer = tf.keras.layers.concatenate(list(inputs.values()))
    layer = tf.keras.layers.Dense(units_1, activation="relu")(layer)
    layer = tf.keras.layers.Dropout(dropout)(layer)
    layer = tf.keras.layers.Dense(units_2, activation="relu")(layer)
    layer = tf.keras.layers.Dropout(dropout)(layer)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def _serving_fn(model, tft_layer, raw_feature_spec):
    """Membuat fungsi serving: tf.Example mentah -> probabilitas kelas."""
    feature_spec = dict(raw_feature_spec)
    feature_spec.pop(LABEL_KEY)

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        parsed = tf.io.parse_example(serialized_tf_examples, feature_spec)
        return model(tft_layer(parsed))

    return serve_tf_examples_fn


def _transform_fn(tft_layer, raw_feature_spec):
    """Membuat fungsi untuk Evaluator: contoh mentah (+ label) -> fitur."""

    @tf.function
    def transform_features_fn(serialized_tf_examples):
        parsed = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        return tft_layer(parsed)

    return transform_features_fn


def save_model(model, tf_transform_output, export_dir):
    """Menyimpan model sebagai SavedModel dengan signature serving.

    Memakai tf.saved_model.save pada tf.Module pembungkus karena
    model.save() di Keras 3 tidak menerima argumen `signatures`.
    """
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    serve_layer = tf_transform_output.transform_features_layer()
    eval_layer = tf_transform_output.transform_features_layer()

    input_spec = tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    signatures = {
        "serving_default": _serving_fn(
            model, serve_layer, raw_feature_spec
        ).get_concrete_function(input_spec),
        "transform_features": _transform_fn(
            eval_layer, raw_feature_spec
        ).get_concrete_function(input_spec),
    }

    export_module = tf.Module()
    export_module.model = model
    export_module.serve_layer = serve_layer
    export_module.eval_layer = eval_layer
    tf.saved_model.save(export_module, export_dir, signatures=signatures)
