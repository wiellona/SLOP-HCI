import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

from app.hand_preprocess import (
    FEATURE_SIZE_ONE_HAND,
    FEATURE_SIZE_TWO_HANDS,
    normalize_hand_landmarks,
    normalize_two_hands,
)

USE_TWO_HANDS = True
REQUIRE_BOTH_HANDS = True
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND

# Inisialisasi MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
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
    hand_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    if USE_TWO_HANDS:
        normalized = normalize_two_hands(
            results.right_hand_landmarks,
            results.left_hand_landmarks,
            require_both_hands=REQUIRE_BOTH_HANDS,
        )
        if normalized is not None:
            hand_features = normalized

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    else:
        if results.right_hand_landmarks:
            normalized = normalize_hand_landmarks(results.right_hand_landmarks)
            if normalized is not None:
                hand_features = normalized

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