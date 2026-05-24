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

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}

USE_TWO_HANDS = _env_flag("SIGN_USE_TWO_HANDS", True)
REQUIRE_BOTH_HANDS = _env_flag("SIGN_REQUIRE_BOTH_HANDS", False)
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND

dataset_path = "data/wlasl_cafe"

# Mencari semua file .mp4 di dalam seluruh sub-folder dataset
video_files = glob.glob(os.path.join(dataset_path, '**', '*.mp4'), recursive=True)

# 2. INISIALISASI MEDIAPIPE DAN DIREKTORI OUTPUT
mp_holistic = mp.solutions.holistic
output_dir = "data/wlasl_npy"

# 3. PROSES ITERASI VIDEO
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
    for video_path in video_files:
        base_name = os.path.basename(video_path)
        file_name, _ = os.path.splitext(base_name)
        parent_dir = os.path.basename(os.path.dirname(video_path))
        
        save_dir = os.path.join(output_dir, parent_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{file_name}.npy")
        
        if os.path.exists(save_path):
            continue

        cap = cv2.VideoCapture(video_path)
        video_features = []

        # 4. EKSTRAKSI FRAME DEMI FRAME
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break 

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            hand_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

            if USE_TWO_HANDS:
                normalized = normalize_two_hands(
                    results.right_hand_landmarks,
                    results.left_hand_landmarks,
                    require_both_hands=REQUIRE_BOTH_HANDS,
                )
            else:
                primary_hand = results.right_hand_landmarks or results.left_hand_landmarks
                normalized = normalize_hand_landmarks(primary_hand)

            if normalized is not None:
                hand_features = normalized

            video_features.append(hand_features)

        cap.release()

        # 5. PENYIMPANAN DATA MATRIKS
        if len(video_features) > 0:
            feature_matrix = np.array(video_features)
            np.save(save_path, feature_matrix)
            print(f"Berhasil mengekstrak: {parent_dir}/{file_name}.mp4 -> Ukuran Matriks: {feature_matrix.shape}")
        else:
            print(f"Gagal memproses/Video kosong: {video_path}")

print("Proses ekstraksi seluruh dataset selesai.")