"""Menjalankan pipeline TFX dengan orchestrator Apache Beam.

Cara pakai:
    python local_pipeline.py
"""

import os
import sys

USERNAME = "aazm_hafidz"  # username Dicoding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, "modules")

# Modul TFX saling mengimpor (transform, model, features), jadi folder
# modules harus ada di sys.path.
sys.path.insert(0, MODULES_DIR)
os.environ["PYTHONPATH"] = MODULES_DIR + os.pathsep + os.environ.get(
    "PYTHONPATH", ""
)

# Modul memakai paket `keras` (Keras 3) langsung agar sama dengan yang dipakai
# KerasTuner. Variabel ini (sering terbawa dari notebook lama) memaksa tf.keras
# menjadi Keras 2 sehingga tipe model tidak cocok, jadi dihapus.
os.environ.pop("TF_USE_LEGACY_KERAS", None)

# pylint: disable=wrong-import-position
from apache_beam.options.pipeline_options import StandardOptions
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from components import init_components

# Beam versi baru memakai PrismRunner (proses gRPC terpisah) sebagai default
# DirectRunner. Pelatihan/tuning yang lama membuat koneksi gRPC-nya timeout
# (DEADLINE_EXCEEDED). FnApiRunner berjalan di dalam proses yang sama dan stabil.
LOCAL_BEAM_RUNNER = "apache_beam.runners.portability.fn_api_runner.FnApiRunner"
# BeamDagRunner membuat beam.Pipeline() tanpa opsi, jadi default runner diganti.
StandardOptions.DEFAULT_RUNNER = LOCAL_BEAM_RUNNER

PIPELINE_NAME = f"{USERNAME}-pipeline"
PIPELINE_DIR = os.path.join(BASE_DIR, PIPELINE_NAME)
PIPELINE_ROOT = os.path.join(PIPELINE_DIR, "pipeline_root")
METADATA_PATH = os.path.join(PIPELINE_DIR, "metadata", "metadata.sqlite")
SERVING_MODEL_DIR = os.path.join(PIPELINE_DIR, "serving_model")
DATA_DIR = os.path.join(BASE_DIR, "data")


def init_local_pipeline(components, pipeline_root):
    """Membungkus komponen menjadi satu pipeline yang berjalan di Beam."""
    beam_args = [
        f"--runner={LOCAL_BEAM_RUNNER}",  # untuk pipeline Beam di dalam komponen
        "--direct_num_workers=1",
    ]
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=(
            metadata.sqlite_metadata_connection_config(METADATA_PATH)
        ),
        beam_pipeline_args=beam_args,
    )


def main():
    """Menyiapkan folder, membuat komponen, lalu menjalankan pipeline."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    os.makedirs(SERVING_MODEL_DIR, exist_ok=True)

    components = init_components(DATA_DIR, MODULES_DIR, SERVING_MODEL_DIR)
    local_pipeline = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=local_pipeline)


if __name__ == "__main__":
    main()
