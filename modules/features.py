"""Konstanta fitur dan konfigurasi bersama untuk dataset Obesity Levels.

`Height` dan `Weight` sengaja tidak dipakai sebagai fitur model karena label
(`NObeyesdad`) diturunkan langsung dari BMI (target leakage).
"""

LABEL_KEY = "NObeyesdad"
NUM_CLASSES = 7

# Fitur kategorikal beserta jumlah kategorinya (di luar 1 bucket OOV).
CATEGORICAL_FEATURES = {
    "Gender": 2,
    "family_history_with_overweight": 2,
    "FAVC": 2,
    "CAEC": 4,
    "SMOKE": 2,
    "SCC": 2,
    "CALC": 4,
    "MTRANS": 5,
}

NUMERICAL_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

BATCH_SIZE = 64


def transformed_name(key):
    """Mengembalikan nama fitur setelah proses Transform."""
    return key + "_xf"
