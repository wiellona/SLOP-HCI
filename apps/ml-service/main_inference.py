import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import json
import os
import time

# Mengimpor arsitektur jaringan saraf Siformer
from app.models.siformer_model import Siformer
from app.hand_preprocess import (
    FEATURE_SIZE_ONE_HAND,
    FEATURE_SIZE_TWO_HANDS,
    normalize_hand_landmarks,
    normalize_two_hands,
    compute_motion_score,
    PredictionFilter,
)

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

LABELS_PATH = "app/models/weights/siformer_labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABEL_MAP = json.load(f)

LABEL_ALIASES = {
    "me": "i",
}

USE_TWO_HANDS = False
REQUIRE_BOTH_HANDS = False
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND
NUM_JOINTS = 42 if USE_TWO_HANDS else 21

# --- 2. KONFIGURASI ARSITEKTUR DAN PEMUATAN PARAMETER ---
NUM_CLASSES = len(LABEL_MAP)
model = Siformer(num_joints=NUM_JOINTS, num_classes=NUM_CLASSES)

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
prev_features = None

MIN_CONFIDENCE = 0.85
SMOOTHING_WINDOW = 5
STABLE_FRAMES = 3
COOLDOWN_FRAMES = 10
MOTION_THRESHOLD = 0.08
NO_HAND_TIMEOUT_SEC = 3.0

prediction_filter = PredictionFilter(
    window_size=SMOOTHING_WINDOW,
    min_confidence=MIN_CONFIDENCE,
    stable_frames=STABLE_FRAMES,
    cooldown_frames=COOLDOWN_FRAMES,
    motion_threshold=MOTION_THRESHOLD,
)

cap = cv2.VideoCapture(0)
print("\n[ STATUS ] Sistem SLOP Daring. Siap menerima input visual dari kamera.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    hand_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    motion_score = None

    if USE_TWO_HANDS:
        normalized = normalize_two_hands(
            results.right_hand_landmarks,
            results.left_hand_landmarks,
            require_both_hands=REQUIRE_BOTH_HANDS,
        )
    else:
        normalized = normalize_hand_landmarks(results.right_hand_landmarks)

    hand_detected = normalized is not None

    if hand_detected:
        hand_features = normalized
        motion_score = compute_motion_score(prev_features, hand_features)
        prev_features = hand_features

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if USE_TWO_HANDS and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        last_hand_time = time.time()
        frame_sequence.append(hand_features)
    else:
        prev_features = None
        prediction_filter.reset()
        frame_sequence.clear()

    # --- 4. PROSES KLASIFIKASI SPASIAL-TEMPORAL ---
    if len(frame_sequence) == SEQUENCE_LENGTH:
        # Dimensi array saat ini: (30, FEATURE_SIZE).
        # Fungsi unsqueeze(0) mengubahnya menjadi Tensor berdimensi: (1, 30, FEATURE_SIZE)
        input_tensor = torch.tensor(np.array(frame_sequence), dtype=torch.float32).unsqueeze(0)

        # Matikan perhitungan gradien secara komprehensif pada level sesi (no_grad)
        with torch.no_grad():
            output_logits = model(input_tensor)
            probabilities = torch.softmax(output_logits, dim=1).cpu().numpy().flatten()

        label_idx, conf, emit = prediction_filter.update(probabilities, motion_score)

        if label_idx is not None:
            predicted_word = LABEL_MAP[label_idx] if label_idx < len(LABEL_MAP) else "Indeks Tidak Valid"
            display_word = LABEL_ALIASES.get(predicted_word, predicted_word)

            if emit:
                if not sentence_words or sentence_words[-1] != display_word:
                    sentence_words.append(display_word)

            display_conf = conf if conf is not None else 0.0
        else:
            display_word = "Menunggu..."
            display_conf = 0.0

        print(
            "Terminal Output | Kata Dideteksi: [ "
            f"{display_word} ] | Conf: {display_conf:.2f}     ",
            end='\r'
        )

        cv2.putText(frame, f"Deteksi AI: {display_word}", (20, 50),
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