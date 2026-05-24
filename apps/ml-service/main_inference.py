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
LABEL_MAP = [
    "allergy", "change", "coffee", "five", "four", "hello", "hot", "me",
    "milk", "no", "one", "order", "peanut butter", "please", "sugar",
    "tea", "thank you", "three", "two", "want", "water", "with",
]

LABELS_PATH = "app/models/weights/siformer_labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABEL_MAP = json.load(f)

TRANSLATION_MAP_PATH = "app/resources/sign_translation_id.json"
TRANSLATION_MAP = {}
if os.path.exists(TRANSLATION_MAP_PATH):
    with open(TRANSLATION_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            TRANSLATION_MAP = {str(k): str(v) for k, v in data.items()}

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}

USE_TWO_HANDS = _env_flag("SIGN_USE_TWO_HANDS", True)
REQUIRE_BOTH_HANDS = _env_flag("SIGN_REQUIRE_BOTH_HANDS", False) 
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND
NUM_JOINTS = 42 if USE_TWO_HANDS else 21

# --- 2. KONFIGURASI ARSITEKTUR DAN PEMUATAN PARAMETER ---
NUM_CLASSES = len(LABEL_MAP)
model = Siformer(num_joints=NUM_JOINTS, num_classes=NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "app/models/weights/siformer_wlasl_cafe.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    if device.type == 'cuda':
        model.half() # Menggunakan presisi FP16 di GPU untuk kecepatan
    print(f"Sistem Berhasil: Matriks parameter dimuat ke {device.type.upper()}")
else:
    raise FileNotFoundError(f"Sistem Gagal: File tidak ditemukan di {weights_path}.")

model.eval()

# --- 3. INISIALISASI MESIN PENGEKSTRAKSI VISUAL ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence_words = []
last_hand_time = time.time()
prev_features = None

MIN_CONFIDENCE = 0.65
SMOOTHING_WINDOW = 5
STABLE_FRAMES = 3
COOLDOWN_FRAMES = 10
MOTION_THRESHOLD = 0.08
NO_HAND_TIMEOUT_SEC = 1.0

prediction_filter = PredictionFilter(
    window_size=SMOOTHING_WINDOW,
    min_confidence=MIN_CONFIDENCE,
    stable_frames=STABLE_FRAMES,
    cooldown_frames=COOLDOWN_FRAMES,
    motion_threshold=MOTION_THRESHOLD,
)

cap = cv2.VideoCapture(0)
print("\n[ STATUS ] Sistem SLOP Daring. Siap menerima input visual dari kamera.")

display_final_text = ""
text_clear_time = 0

# Variabel untuk menampung UI deteksi kata saat ini (Real-time)
current_word_display = ""
current_conf_display = 0.0

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
        primary_hand = results.right_hand_landmarks or results.left_hand_landmarks
        normalized = normalize_hand_landmarks(primary_hand)

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
        display_final_text = "" 
    else:
        prev_features = None
        prediction_filter.reset()
        frame_sequence.clear()
        current_word_display = ""
        current_conf_display = 0.0

    # --- 4. PROSES KLASIFIKASI SPASIAL-TEMPORAL ---
    current_status = "Menunggu..."
    
    if len(frame_sequence) == SEQUENCE_LENGTH:
        input_tensor = torch.tensor(np.array(frame_sequence), dtype=torch.float32).unsqueeze(0).to(device)
        if device.type == 'cuda':
            input_tensor = input_tensor.half()

        with torch.no_grad():
            output_logits = model(input_tensor)
            probabilities = torch.softmax(output_logits, dim=1).cpu().numpy().flatten()

        label_idx, conf, emit = prediction_filter.update(probabilities, motion_score)

        if label_idx is not None:
            predicted_word = LABEL_MAP[label_idx] if label_idx < len(LABEL_MAP) else "Indeks Tidak Valid"
            translated_word = TRANSLATION_MAP.get(predicted_word, predicted_word)

            # Update variabel untuk ditampilkan di layar (real-time)
            current_word_display = translated_word
            current_conf_display = conf if conf is not None else 0.0

            if emit:
                if not sentence_words or sentence_words[-1] != translated_word:
                    sentence_words.append(translated_word)

        current_status = "Membaca gerakan..."
    elif hand_detected:
        current_status = "Mengumpulkan frame..."

    # --- 5. LOGIKA OUTPUT (TIMEOUT) ---
    if not hand_detected and (time.time() - last_hand_time) >= NO_HAND_TIMEOUT_SEC:
        if sentence_words:
            final_sentence = " ".join(sentence_words)
            print(f"\n[ SLOP Output ] Hasil Terjemahan: {final_sentence}")
            
            display_final_text = final_sentence 
            text_clear_time = time.time() + 3.0 
            
            sentence_words.clear()
        
        last_hand_time = time.time()

    # --- 6. VISUALISASI ANTARMUKA ---
    # Skenario 1: Menampilkan kalimat akhir saat tangan sudah turun
    if display_final_text and time.time() < text_clear_time:
        cv2.putText(frame, f"Output Akhir: {display_final_text}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Skenario 2: Menampilkan tracking saat tangan sedang di depan kamera
    elif hand_detected and current_word_display:
        # Penentuan warna berdasarkan confidence (Format BGR OpenCV)
        if current_conf_display >= 0.85:
            text_color = (0, 255, 0)      # Hijau (Sangat Yakin)
        elif current_conf_display >= 0.60:
            text_color = (0, 255, 255)    # Kuning (Ragu-ragu)
        else:
            text_color = (0, 0, 255)      # Merah (Tidak Yakin)

        # Menampilkan teks tebakan kata saat ini + confidence score
        cv2.putText(frame, f"Prediksi: {current_word_display} ({current_conf_display:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        
        # Menampilkan draft kalimat yang sudah tersusun di bawah layar (jika ada)
        if sentence_words:
            draft_text = "Draft: " + " ".join(sentence_words)
            cv2.putText(frame, draft_text, (20, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Skenario 3: Menampilkan status standar
    else:
        cv2.putText(frame, f"Status: {current_status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('SLOP - Sistem Inferensi Waktu Nyata', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()