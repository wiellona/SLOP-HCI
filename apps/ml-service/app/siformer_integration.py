import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

# Inisialisasi MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_iqge_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 1. SETUP STRUKTUR DATA TEMPORAL
# Menggunakan deque (Double-Ended Queue) dengan batas maksimal 30.
# Secara otomatis, jika frame ke-31 masuk, frame ke-1 akan terhapus.
# Ini menciptakan "jendela waktu" (sliding window) yang terus bergerak.
SEQUENCE_LENGTH = 30
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
print("Sistem Sekuens Dimulai... Kumpulkan 30 frame untuk membentuk 1 Tensor.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # 2. INISIALISASI ARRAY UNTUK 1 FRAME
    # Membuat array kosong berisi angka nol sebanyak 63 (21 titik * 3 sumbu koordinat)
    hand_features = np.zeros(63)

    if results.right_hand_landmarks:
        # 3. NORMALISASI SPASIAL
        # Mengambil koordinat pergelangan tangan (wrist) pada indeks 0 sebagai pusat referensi
        wrist = results.right_hand_landmarks.landmark[0]
        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z

        # Melakukan iterasi (perulangan) pada 21 titik tangan
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            # Normalisasi: Koordinat titik saat ini dikurangi koordinat wrist
            norm_x = landmark.x - wrist_x
            norm_y = landmark.y - wrist_y
            norm_z = landmark.z - wrist_z

            # Memasukkan nilai hasil normalisasi ke dalam array yang tepat.
            # Rumus (i*3) memastikan X, Y, Z tersusun urut: x0, y0, z0, x1, y1, z1, dst.
            hand_features[i*3] = norm_x
            hand_features[(i*3) + 1] = norm_y
            hand_features[(i*3) + 2] = norm_z

        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 4. PENYIMPANAN KE DALAM ANTREAN
    # Nilai 63 koordinat (baik itu 0 semua karena tidak terdeteksi, atau hasil perhitungan)
    # dimasukkan ke dalam antrean memory.
    frame_sequence.append(hand_features)

    # 5. PEMBENTUKAN TENSOR SIFORMER
    # Jika antrean sudah mengumpulkan tepat 30 frame
    if len(frame_sequence) == SEQUENCE_LENGTH:
        # Mengonversi antrean menjadi bentuk matriks NumPy: ukuran (30, 63)
        sequence_matrix = np.array(frame_sequence)
        
        # Mengonversi NumPy menjadi format Tensor PyTorch dan 
        # menambahkan dimensi Batch menggunakan unsqueeze(0)
        # Tensor ini yang secara teknis akan diumpankan ke model Siformer
        input_tensor = torch.tensor(sequence_matrix, dtype=torch.float32).unsqueeze(0)
        
        # Mencetak ukuran matriks ke terminal untuk membuktikan kebenaran strukturnya
        print(f"Tensor Siap! Ukuran Dimensi Matriks: {list(input_tensor.shape)}")

    cv2.imshow('SLOP - Tensor Preparation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()