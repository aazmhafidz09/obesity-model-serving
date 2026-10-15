"""Modul Tuner: pencarian hyperparameter otomatis dengan KerasTuner."""

import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult

from model import build_model, input_fn

MAX_TRIALS = 10
EPOCHS = 10
STEPS_PER_EPOCH = 50
OBJECTIVE = "val_sparse_categorical_accuracy"


def tuner_fn(fn_args):
    """Menyiapkan tuner yang dijalankan oleh komponen TFX Tuner."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_set = input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective=keras_tuner.Objective(OBJECTIVE, direction="max"),
        max_trials=MAX_TRIALS,
        directory=fn_args.working_dir,
        project_name="obesity_tuning",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=OBJECTIVE, mode="max", patience=3
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "validation_steps": fn_args.eval_steps,
            "callbacks": [early_stopping],
        },
    )
