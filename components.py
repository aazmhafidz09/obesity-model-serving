"""Definisi seluruh komponen TFX untuk pipeline klasifikasi obesitas."""

import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

LABEL_KEY_TRANSFORMED = "NObeyesdad_xf"


def _eval_config(accuracy_threshold):
    """Konfigurasi TFMA: metrik dan threshold agar model di-bless."""
    return tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key=LABEL_KEY_TRANSFORMED,
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="SparseCategoricalAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": accuracy_threshold}
                            ),
                        ),
                    ),
                ]
            )
        ],
    )


def init_components(data_dir, modules_dir, serving_model_dir,
                    train_steps=1000, eval_steps=200,
                    accuracy_threshold=0.75):
    """Membuat semua komponen pipeline.

    Args:
        data_dir: folder berisi file CSV dataset.
        modules_dir: folder berisi transform.py, tuner.py, dan trainer.py.
        serving_model_dir: tujuan Pusher untuk menyimpan model.
        train_steps: jumlah langkah pelatihan.
        eval_steps: jumlah langkah evaluasi.
        accuracy_threshold: batas minimal akurasi agar model di-bless.

    Returns:
        Tuple komponen sesuai urutan eksekusi.
    """
    split_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_dir, output_config=split_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(modules_dir, "transform.py"),
    )

    train_args = trainer_pb2.TrainArgs(splits=["train"], num_steps=train_steps)
    eval_args = trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps)

    tuner = Tuner(
        module_file=os.path.join(modules_dir, "tuner.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=train_args,
        eval_args=eval_args,
    )

    trainer = Trainer(
        module_file=os.path.join(modules_dir, "trainer.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=train_args,
        eval_args=eval_args,
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=_eval_config(accuracy_threshold),
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )
