import os
import glob
import cv2
import mediapipe as mp
import numpy as np

from app.hand_preprocess import (
    FEATURE_SIZE_ONE_HAND,
    FEATURE_SIZE_TWO_HANDS,
    normalize_hand_landmarks,
    normalize_two_hands,
)

USE_TWO_HANDS = False
REQUIRE_BOTH_HANDS = False
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND

# 1. MENGAKSES LOKASI DATASET
print("Membaca direktori dataset lokal...")
dataset_path = "data/wlasl_cafe"
print(f"Dataset ditemukan di: {dataset_path}")

# Mencari semua file .mp4 di dalam seluruh sub-folder dataset
video_files = glob.glob(os.path.join(dataset_path, '**', '*.mp4'), recursive=True)
print(f"Total video yang akan diproses: {len(video_files)}")

# 2. INISIALISASI MEDIAPIPE DAN DIREKTORI OUTPUT
mp_holistic = mp.solutions.holistic
output_dir = "data/wlasl_npy"

# 3. PROSES ITERASI VIDEO
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
    for video_path in video_files:
        # Mengambil nama file video tanpa ekstensi untuk penamaan file output
        base_name = os.path.basename(video_path)
        file_name, _ = os.path.splitext(base_name)
        
        # Mengekstrak nama folder induk untuk mengetahui kelas kata (misal: "Halo", "Maaf")
        parent_dir = os.path.basename(os.path.dirname(video_path))
        
        # Membuat folder output berdasarkan kelas kata
        save_dir = os.path.join(output_dir, parent_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{file_name}.npy")
        
        # Jika file .npy sudah ada, lewati proses untuk menghemat waktu komputasi
        if os.path.exists(save_path):
            continue

        cap = cv2.VideoCapture(video_path)
        video_features = []

        # 4. EKSTRAKSI FRAME DEMI FRAME
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break # Video selesai

            # Konversi format warna BGR ke RGB untuk MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Inisialisasi matriks nol berukuran 63 (21 titik * 3 sumbu)
            hand_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

            # Normalisasi spasial dengan menjadikan pergelangan tangan (indeks 0) sebagai titik pusat
            if USE_TWO_HANDS:
                normalized = normalize_two_hands(
                    results.right_hand_landmarks,
                    results.left_hand_landmarks,
                    require_both_hands=REQUIRE_BOTH_HANDS,
                )
            else:
                normalized = normalize_hand_landmarks(results.right_hand_landmarks)

            if normalized is not None:
                hand_features = normalized

            # Menambahkan matriks koordinat dari frame ini ke dalam urutan video
            video_features.append(hand_features)

        cap.release()

        # 5. PENYIMPANAN DATA MATRIKS
        if len(video_features) > 0:
            # Mengonversi urutan frame menjadi matriks NumPy
            feature_matrix = np.array(video_features)
            # Menyimpan matriks ke dalam file biner .npy
            np.save(save_path, feature_matrix)
            print(f"Berhasil mengekstrak: {parent_dir}/{file_name}.mp4 -> Ukuran Matriks: {feature_matrix.shape}")
        else:
            print(f"Gagal memproses/Video kosong: {video_path}")

print("Proses ekstraksi seluruh dataset selesai.")