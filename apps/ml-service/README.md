## 🛠️ Persiapan Awal

1. **Buat Virtual Environment:** `python -m venv venv`
2. **Aktivasi venv:** `.\venv\Scripts\activate` (PowerShell)
3. **Install Dependensi:** `pip install -r requirements.txt`

## 📂 Fase 1: Akuisisi Data

Untuk mendapatkan dataset, jalankan skrip `setup_wlasl.py`. Skrip ini akan mengunduh dataset dari Kaggle dan memfilter 15 kata kunci (Coffee Shop context).

```bash
python setup_wlasl.py
```

## 🎥 Fase 2: Preprocessing (Lokal)

Ubah video mentah (.mp4) menjadi matriks koordinat (.npy) menggunakan MediaPipe. Proses ini dilakukan di laptop (CPU).

```bash
python preprocess_videos.py
```

Konfigurasi dua tangan (opsional) dapat diatur lewat environment:

```bash
set SIGN_USE_TWO_HANDS=true
set SIGN_REQUIRE_BOTH_HANDS=false
```

Hasilnya akan ada di folder data/wlasl_npy/. Setelah selesai, ZIP folder tersebut bersama file train.py dan folder app/ untuk diunggah ke Colab.

## 🧠 Fase 3: Training (Google Colab)

Karena training membutuhkan GPU besar, gunakan Google Colab. Berikut urutan kodenya:

Sel 1: Hubungkan Drive

```py
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/SLOP_Project')
```

Sel 2: Ekstrasi Dataa

```py
!unzip -q wlasl_training.zip -d /content/wlasl_colab
```

Sel 3: Setup Modul & Library

```py
%cd /content/wlasl_colab
!touch app/__init__.py
!pip install -q mediapipe torch numpy
```

Sel 4: Jalankan Training

```py
!PYTHONPATH=. python train.py
```

Hyperparameter dan fine-tuning minimal dapat diatur lewat environment:

```bash
set SIGN_USE_TWO_HANDS=true
set SIGN_FREEZE_ENCODER=true
set SIGN_PRETRAINED_PATH=app/models/weights/siformer_wlasl_cafe.pth
set SIGN_EPOCHS=50
set SIGN_LR=0.001
```

## 📊 Benchmark & Metrik

Gunakan skrip benchmark untuk mengecek akurasi dan latency:

```bash
python scripts/benchmark_sign.py --data-root data/wlasl_npy
```

## 🚀 Fase 4: Real-time Inference

Setelah training selesai, download file siformer_wlasl_cafe.pth dari Colab ke app/models/weights/. Jalankan sistem kasir:

```bash
python main_inference.py
```
