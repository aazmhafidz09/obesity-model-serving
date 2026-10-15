# Submission 2: Klasifikasi Tingkat Obesitas (TFX, Railway, Prometheus)

Nama: Aaz Muhammad Hafidz Azis

Username dicoding: aazm_hafidz

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) (UCI ML Repository, ±2.111 baris, 16 fitur, 7 kelas). 23% data dikumpulkan lewat survei, 77% dibangkitkan dengan SMOTE. |
| Masalah | Skrining tingkat berat badan/obesitas umumnya memerlukan pengukuran fisik (BMI). Proyek ini menguji apakah tingkat berat badan dapat diprediksi hanya dari kebiasaan makan dan kondisi fisik, tanpa tinggi dan berat badan. |
| Solusi machine learning | Model klasifikasi 7 kelas dalam TFX pipeline yang berjalan di Apache Beam, lalu disajikan sebagai REST API lewat TensorFlow Serving di cloud. Target: akurasi evaluasi ≥ 75%. |
| Metode pengolahan | Split data 80:20. Fitur numerik (`Age, FCVC, NCP, CH2O, FAF, TUE`) distandardisasi (z-score). Fitur kategorikal di-*vocabulary* lalu one-hot (+1 bucket OOV). Label diubah menjadi integer. `Height` dan `Weight` dibuang karena label diturunkan dari BMI (*target leakage*). |
| Arsitektur model | Neural network feed-forward: `Dense(units_1) → Dropout → Dense(units_2) → Dropout → Dense(7, softmax)`. `units_1`, `units_2`, `dropout`, dan `learning_rate` dicari otomatis oleh komponen Tuner (KerasTuner RandomSearch, 10 trial). Optimizer Adam, loss `SparseCategoricalCrossentropy`. |
| Metrik evaluasi | `SparseCategoricalAccuracy` (threshold ≥ 0,75) dan `ExampleCount`, dihitung oleh TFMA pada komponen Evaluator. |
| Performa model | _TODO: isi setelah pipeline dijalankan: hyperparameter terbaik, akurasi latih, akurasi evaluasi, status blessed._ |
| Opsi deployment | _TODO: contoh: model disajikan dengan TensorFlow Serving di dalam container Docker dan di-deploy ke Railway (pengganti Heroku, yang tidak lagi gratis). Alasan: mendukung Dockerfile, mudah diatur lewat GitHub, tersedia domain HTTPS publik._ |
| Web app | [obesity-model](https://obesity-model-serving-production.up.railway.app/v1/models/obesity-model/metadata) |
| Monitoring | _TODO: ringkas hasil Prometheus: jumlah request, latensi, status 200 vs error, dan apa yang terlihat setelah request uji dikirim._ |

## Struktur Proyek

```
submission-2/
├── <username>-pipeline/          # artefak pipeline (dibuat saat pipeline dijalankan)
│   ├── pipeline_root/
│   ├── metadata/
│   └── serving_model/
├── serving_model/                # salinan serving_model untuk Docker build
├── modules/                      # features, transform, model, tuner, trainer
├── monitoring/                   # Dockerfile, prometheus.config, prometheus.yml
├── components.py                 # definisi komponen TFX
├── local_pipeline.py             # menjalankan pipeline dengan Apache Beam
├── pipeline.ipynb                # notebook yang sudah dijalankan
├── <username>-testing.ipynb      # uji prediction request ke cloud
├── Dockerfile                    # TF Serving + metrik Prometheus
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Pasang dependency: `pip install -r requirements.txt` (di environment baru; di Colab cukup jalankan notebook).
2. Taruh CSV dataset di `data/`, lalu jalankan `python local_pipeline.py` (atau sel di `pipeline.ipynb`).
3. Salin `<username>-pipeline/serving_model/` ke `serving_model/` di root.
4. Build dan uji image secara lokal:
   ```
   docker build -t obesity-model .
   docker run -p 8501:8501 obesity-model
   ```
5. Deploy ke Railway, lalu uji dengan `<username>-testing.ipynb`.
6. Deploy folder `monitoring/` sebagai layanan Prometheus terpisah.
7. Skor kualitas kode: `pylint modules`.
