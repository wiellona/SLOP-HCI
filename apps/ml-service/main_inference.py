import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import os
import time

# Mengimpor arsitektur jaringan saraf Siformer
from app.models.siformer_model import Siformer

# --- 1. DEKLARASI PEMETAAN KELAS (LABEL MAPPING) ---
# Urutan label harus sama dengan urutan folder kelas yang dipakai saat training (abjad).
LABEL_MAP = [
    "coffee",
    "five",
    "four",
    "hello",
    "hot",
    "milk",
    "no",
    "one",
    "order",
    "please",
    "sugar",
    "tea",
    "thank you",
    "three",
    "two",
    "want",
    "water",
]

# --- 2. KONFIGURASI ARSITEKTUR DAN PEMUATAN PARAMETER ---
NUM_CLASSES = len(LABEL_MAP)
model = Siformer(num_classes=NUM_CLASSES)

weights_path = "app/models/weights/siformer_wlasl_cafe.pth"

# Memeriksa eksistensi file matriks secara fisik di dalam sistem penyimpanan
if os.path.exists(weights_path):
    # Mengalokasikan matriks parameter ke memori CPU
    # CPU digunakan karena proses inferensi tunggal real-time tidak menuntut pemrosesan paralel masif seperti GPU
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    print(f"Sistem Berhasil: Matriks parameter dimuat dari {weights_path}")
else:
    raise FileNotFoundError(f"Sistem Gagal: File tidak ditemukan di {weights_path}. Periksa kembali hierarki folder.")

# Mengunci lapisan arsitektur model dari modifikasi matematis
model.eval()

# --- 3. INISIALISASI MESIN PENGEKSTRAKSI VISUAL ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence_words = []
last_hand_time = time.time()

CONFIDENCE_THRESHOLD = 0.8
NO_HAND_TIMEOUT_SEC = 3.0

cap = cv2.VideoCapture(0)
print("\n[ STATUS ] Sistem SLOP Daring. Siap menerima input visual dari kamera.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    hand_features = np.zeros(63)

    hand_detected = results.right_hand_landmarks is not None
    if hand_detected:
        wrist = results.right_hand_landmarks.landmark[0]
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            hand_features[i*3] = landmark.x - wrist.x
            hand_features[(i*3) + 1] = landmark.y - wrist.y
            hand_features[(i*3) + 2] = landmark.z - wrist.z
            
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        last_hand_time = time.time()

    frame_sequence.append(hand_features)

    # --- 4. PROSES KLASIFIKASI SPASIAL-TEMPORAL ---
    if len(frame_sequence) == SEQUENCE_LENGTH:
        # Dimensi array saat ini: (30, 63). 
        # Fungsi unsqueeze(0) mengubahnya menjadi Tensor berdimensi: (1, 30, 63)
        input_tensor = torch.tensor(np.array(frame_sequence), dtype=torch.float32).unsqueeze(0)

        # Matikan perhitungan gradien secara komprehensif pada level sesi (no_grad)
        with torch.no_grad():
            output_logits = model(input_tensor)
            probabilities = torch.softmax(output_logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
            confidence_value = confidence.item()
            predicted_index = predicted_index.item()
        
        if predicted_index < len(LABEL_MAP):
            predicted_word = LABEL_MAP[predicted_index]
        else:
            predicted_word = "Indeks Tidak Valid"

        if hand_detected and confidence_value >= CONFIDENCE_THRESHOLD:
            if not sentence_words or sentence_words[-1] != predicted_word:
                sentence_words.append(predicted_word)

        print(
            "Terminal Output | Kata Dideteksi: [ "
            f"{predicted_word} ] | Indeks Node: {predicted_index} | "
            f"Conf: {confidence_value:.2f}     ",
            end='\r'
        )

        cv2.putText(frame, f"Deteksi AI: {predicted_word}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    if not hand_detected and (time.time() - last_hand_time) >= NO_HAND_TIMEOUT_SEC:
        if sentence_words:
            print("\nKalimat: " + " ".join(sentence_words))
            sentence_words.clear()
        last_hand_time = time.time()

    cv2.imshow('SLOP - Sistem Inferensi Waktu Nyata', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()