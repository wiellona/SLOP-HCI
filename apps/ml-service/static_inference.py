import json
import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from typing import List

from app.hand_preprocess import (
    FEATURE_SIZE_ONE_HAND,
    FEATURE_SIZE_TWO_HANDS,
    PredictionFilter,
    compute_motion_score,
    normalize_hand_landmarks,
    normalize_two_hands,
)

MIN_CONFIDENCE = 0.85
SMOOTHING_WINDOW = 5
STABLE_FRAMES = 3
COOLDOWN_FRAMES = 10
MOTION_THRESHOLD = 0.08
NO_HAND_TIMEOUT_SEC = 1.0

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


USE_TWO_HANDS = _env_flag("SIGN_USE_TWO_HANDS", False) # Set to True if your model was trained with two-hand features
REQUIRE_BOTH_HANDS = _env_flag("SIGN_REQUIRE_BOTH_HANDS", False)
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND


def load_labels(labels_path: str) -> List[str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "app", "models", "weights", "static_svm.joblib")
    labels_path = os.path.join(base_dir, "app", "models", "weights", "static_labels.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Static model or labels not found. Run train_static_classifier.py first.")

    model = joblib.load(model_path)
    labels = load_labels(labels_path)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    prediction_filter = PredictionFilter(
        window_size=SMOOTHING_WINDOW,
        min_confidence=MIN_CONFIDENCE,
        stable_frames=STABLE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
        motion_threshold=MOTION_THRESHOLD,
    )

    prev_features = None
    sentence_words = []
    last_hand_time = time.time()

    cap = cv2.VideoCapture(0)
    print("[ STATUS ] Static gesture inference ready.")

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
        else:
            prev_features = None
            prediction_filter.reset()

        if hand_detected:
            probs = model.predict_proba(hand_features.reshape(1, -1))[0]
            label_idx, conf, emit = prediction_filter.update(probs, motion_score)

            if label_idx is not None:
                predicted_word = labels[label_idx] if label_idx < len(labels) else "Unknown"
                if emit:
                    if not sentence_words or sentence_words[-1] != predicted_word:
                        sentence_words.append(predicted_word)

                display_word = " ".join(sentence_words) if sentence_words else predicted_word
                display_conf = conf if conf is not None else 0.0
            else:
                display_word = "Menunggu..."
                display_conf = 0.0

            print(
                "Terminal Output | Kalimat Sementara: [ "
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

        cv2.imshow("SLOP - Static Gesture Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    main()
