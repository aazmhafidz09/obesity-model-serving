"""Modul Trainer: melatih model dengan hyperparameter terbaik dari Tuner."""

import keras
import keras_tuner
import tensorflow_transform as tft

from model import build_model, input_fn, save_model

EPOCHS = 30
PATIENCE = 5
MONITOR = "val_sparse_categorical_accuracy"


def run_fn(fn_args):
    """Fungsi utama yang dipanggil oleh komponen TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_set = input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    model = build_model(hparams)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR, mode="max", patience=PATIENCE,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_set,
        validation_data=eval_set,
        epochs=EPOCHS,
        steps_per_epoch=fn_args.train_steps // 20,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )

    save_model(model, tf_transform_output, fn_args.serving_model_dir)
