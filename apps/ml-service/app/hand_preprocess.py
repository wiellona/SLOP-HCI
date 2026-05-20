from collections import deque
from typing import Optional, Tuple

import numpy as np

LANDMARK_COUNT = 21
FEATURE_SIZE_ONE_HAND = LANDMARK_COUNT * 3
FEATURE_SIZE_TWO_HANDS = LANDMARK_COUNT * 2 * 3
FEATURE_SIZE = FEATURE_SIZE_ONE_HAND


def _landmarks_to_array(hand_landmarks: Optional[object]) -> Optional[np.ndarray]:
    if hand_landmarks is None:
        return None

    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    if coords.size == 0:
        return None

    return coords


def normalize_hand_landmarks(hand_landmarks: Optional[object], eps: float = 1e-6) -> Optional[np.ndarray]:
    coords = _landmarks_to_array(hand_landmarks)
    if coords is None:
        return None

    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale < eps:
        scale = eps

    coords = coords / scale
    return coords.reshape(-1)


def normalize_two_hands(
    right_hand_landmarks: Optional[object],
    left_hand_landmarks: Optional[object],
    eps: float = 1e-6,
    require_both_hands: bool = False,
) -> Optional[np.ndarray]:
    right = _landmarks_to_array(right_hand_landmarks)
    left = _landmarks_to_array(left_hand_landmarks)

    if right is None and left is None:
        return None

    if require_both_hands and (right is None or left is None):
        return None

    if right is not None and left is not None:
        origin = (right[0] + left[0]) / 2.0
        right = right - origin
        left = left - origin
        combined = np.vstack([right, left])
        scale = np.max(np.linalg.norm(combined, axis=1))
        if scale < eps:
            scale = eps
        combined = combined / scale
        return combined.reshape(-1)

    if right is not None:
        right = right - right[0]
        scale = np.max(np.linalg.norm(right, axis=1))
        if scale < eps:
            scale = eps
        right = right / scale
        left = np.zeros_like(right)
        return np.vstack([right, left]).reshape(-1)

    left = left - left[0]
    scale = np.max(np.linalg.norm(left, axis=1))
    if scale < eps:
        scale = eps
    left = left / scale
    right = np.zeros_like(left)
    return np.vstack([right, left]).reshape(-1)


def compute_motion_score(
    prev_features: Optional[np.ndarray],
    curr_features: Optional[np.ndarray],
) -> float:
    if prev_features is None or curr_features is None:
        return 0.0

    diff = np.abs(curr_features - prev_features)
    return float(np.mean(diff))


class PredictionFilter:
    def __init__(
        self,
        window_size: int = 5,
        min_confidence: float = 0.8,
        stable_frames: int = 3,
        cooldown_frames: int = 10,
        motion_threshold: Optional[float] = None,
    ) -> None:
        self.window = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.cooldown_frames = cooldown_frames
        self.motion_threshold = motion_threshold
        self.last_label: Optional[int] = None
        self.stable_count = 0
        self.cooldown = 0

    def reset(self) -> None:
        self.window.clear()
        self.last_label = None
        self.stable_count = 0
        self.cooldown = 0

    def _tick_cooldown(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1

    def update(
        self,
        probs: Optional[np.ndarray],
        motion_score: Optional[float] = None,
    ) -> Tuple[Optional[int], Optional[float], bool]:
        if probs is None:
            self._tick_cooldown()
            self.stable_count = 0
            return None, None, False

        if self.motion_threshold is not None and motion_score is not None:
            if motion_score > self.motion_threshold:
                self._tick_cooldown()
                self.stable_count = 0
                return None, None, False

        self.window.append(probs)
        if len(self.window) < self.window.maxlen:
            self._tick_cooldown()
            return None, None, False

        avg_probs = np.mean(self.window, axis=0)
        label = int(np.argmax(avg_probs))
        conf = float(avg_probs[label])

        if conf < self.min_confidence:
            self.last_label = label
            self.stable_count = 0
            self._tick_cooldown()
            return label, conf, False

        if label == self.last_label:
            self.stable_count += 1
        else:
            self.last_label = label
            self.stable_count = 1

        self._tick_cooldown()

        if self.cooldown > 0:
            return label, conf, False

        if self.stable_count >= self.stable_frames:
            self.cooldown = self.cooldown_frames
            return label, conf, True

        return label, conf, False
