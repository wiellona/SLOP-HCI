# SLOP ML Service — Technical Implementation Guide
## `apps/ml-service` | Python · FastAPI · Siformer · Whisper

---

| Field | Value |
|---|---|
| **Document Version** | 1.0.0 |
| **Role** | Senior Machine Learning Engineer & Computer Vision Architect |
| **Target Service** | `apps/ml-service` (Python / FastAPI) |
| **Primary Language** | Indonesian Sign Language (BISINDO) |
| **Date** | April 2026 |

---

## Table of Contents

1. [Service Overview & Architecture](#1-service-overview--architecture)
2. [Environment Setup & Dependencies](#2-environment-setup--dependencies)
3. [Data Strategy & Preprocessing Pipeline](#3-data-strategy--preprocessing-pipeline)
4. [Siformer: Spatial-Temporal Transformer](#4-siformer-spatial-temporal-transformer)
5. [Voice Engine: Whisper Integration](#5-voice-engine-whisper-integration)
6. [FastAPI Streaming Architecture](#6-fastapi-streaming-architecture)
7. [Confidence Score & Occlusion Detection](#7-confidence-score--occlusion-detection)
8. [RLHF Export & Fine-Tuning Loop](#8-rlhf-export--fine-tuning-loop)
9. [Socket.io Event Contract Implementation](#9-socketio-event-contract-implementation)
10. [Testing & Benchmarking](#10-testing--benchmarking)
11. [Deployment & GPU Configuration](#11-deployment--gpu-configuration)

---

## 1. Service Overview & Architecture

The `ml-service` is the AI inference backbone of SLOP. It operates as a standalone **Python / FastAPI** microservice responsible for:

- **Sign Language Recognition (SLR):** Real-time 30 FPS skeletal landmark extraction via MediaPipe → temporal sequence encoding via **Siformer** (Spatial-Temporal Transformer)
- **Speech-to-Text (STT):** Audio chunk transcription via **OpenAI Whisper** fine-tuned for Indonesian (BISINDO context)
- **RLHF Pipeline:** Push-back mechanism to the Node.js server, and automated pull of `GOLD`/`SILVER` tier correction data for fine-tuning triggers

### 1.1 Internal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ml-service                               │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  WebSocket  │    │   REST Endpoints  │    │  Background   │  │
│  │  /v1/sign/  │    │  /v1/voice/       │    │  RLHF Task    │  │
│  │  stream     │    │  /v1/health       │    │  Scheduler    │  │
│  └──────┬──────┘    └────────┬─────────┘    └───────┬───────┘  │
│         │                   │                        │          │
│  ┌──────▼──────────────────▼────────────────────────▼───────┐  │
│  │                    Service Layer                          │  │
│  │                                                           │  │
│  │  ┌──────────────────┐      ┌────────────────────────┐    │  │
│  │  │  SignInference   │      │   VoiceInference       │    │  │
│  │  │  Service         │      │   Service              │    │  │
│  │  │                  │      │                        │    │  │
│  │  │ MediaPipe        │      │ Whisper large-v3       │    │  │
│  │  │ → Siformer       │      │ (Indonesian fine-tune) │    │  │
│  │  │ → Confidence     │      │ → Streaming decoder    │    │  │
│  │  │ → Occlusion      │      │                        │    │  │
│  │  └──────────────────┘      └────────────────────────┘    │  │
│  │                                                           │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │              SocketIO Emitter                     │   │  │
│  │  │   inference_started → text_streamed →             │   │  │
│  │  │   inference_complete                              │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
   Node.js Server                        PostgreSQL
   (Socket.io relay)                  (via Node REST)
```

### 1.2 Folder Structure

```
apps/ml-service/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── sign.py              # SLR WebSocket + REST endpoints
│   │       ├── voice.py             # Whisper transcription endpoints
│   │       └── rlhf.py              # RLHF correction + export endpoints
│   ├── core/
│   │   ├── config.py                # Pydantic Settings
│   │   ├── events.py                # App startup/shutdown lifecycle
│   │   └── logging.py               # Structured logging setup
│   ├── models/
│   │   ├── siformer/
│   │   │   ├── __init__.py
│   │   │   ├── architecture.py      # Transformer model definition
│   │   │   ├── loader.py            # Weight loading + device placement
│   │   │   └── weights/             # .pt checkpoint files (gitignored)
│   │   └── whisper/
│   │       ├── __init__.py
│   │       ├── loader.py            # Whisper model loader
│   │       └── fine_tuned/          # Fine-tuned .pt weights (gitignored)
│   ├── schemas/
│   │   ├── sign.py                  # Pydantic I/O schemas for SLR
│   │   ├── voice.py                 # Pydantic I/O schemas for STT
│   │   └── rlhf.py                  # RLHF payload schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sign_inference.py        # MediaPipe + Siformer orchestration
│   │   ├── voice_inference.py       # Whisper inference orchestration
│   │   ├── socketio_emitter.py      # Node.js socket event emitter
│   │   ├── landmark_processor.py    # Coordinate extraction & normalization
│   │   ├── occlusion_detector.py    # Hand visibility + occlusion logic
│   │   └── rlhf_scheduler.py        # Background fine-tune trigger
│   └── main.py                      # FastAPI application factory
├── data/
│   ├── raw/                         # Raw video datasets (gitignored)
│   ├── processed/                   # Normalized .npy skeleton sequences
│   └── label_maps/
│       └── bisindo_labels.json      # Gloss → class index mapping
├── scripts/
│   ├── preprocess_dataset.py        # Offline preprocessing pipeline
│   ├── train_siformer.py            # Training entry point
│   ├── evaluate_model.py            # Evaluation on test split
│   └── export_onnx.py               # Export to ONNX for production
├── tests/
│   ├── test_sign_inference.py
│   ├── test_voice_inference.py
│   ├── test_landmark_processor.py
│   └── conftest.py
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## 2. Environment Setup & Dependencies

### 2.1 `requirements.txt`

```txt
# ── Web Framework ──────────────────────────────────────────────
fastapi==0.111.0
uvicorn[standard]==0.29.0
websockets==12.0
python-socketio==5.11.2       # Socket.io client to emit to Node server
python-multipart==0.0.9       # For audio file uploads

# ── AI / ML Core ───────────────────────────────────────────────
torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0
openai-whisper==20240930
transformers==4.41.0          # HuggingFace for fine-tuning utilities

# ── Computer Vision ─────────────────────────────────────────────
mediapipe==0.10.14
opencv-python-headless==4.9.0.80
Pillow==10.3.0

# ── Data Processing ─────────────────────────────────────────────
numpy==1.26.4
scipy==1.13.0
einops==0.8.0                 # Tensor rearrangement for Transformer
scikit-learn==1.4.2

# ── Validation & Config ─────────────────────────────────────────
pydantic==2.7.1
pydantic-settings==2.2.1

# ── HTTP & Async ────────────────────────────────────────────────
httpx==0.27.0                 # Async HTTP client for Node.js push-back
aiofiles==23.2.1

# ── Observability ───────────────────────────────────────────────
structlog==24.1.0
prometheus-fastapi-instrumentator==7.0.0

# ── Testing ──────────────────────────────────────────────────────
pytest==8.2.0
pytest-asyncio==0.23.6
httpx==0.27.0                 # Also used for TestClient
```

### 2.2 `core/config.py`

```python
# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Service Identity ───────────────────────────────────────────
    service_name: str = "slop-ml-service"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── Node.js Integration ────────────────────────────────────────
    node_server_url: str = "http://server:4000"
    ml_service_api_key: str                      # Shared secret for auth

    # ── Siformer ──────────────────────────────────────────────────
    siformer_model_path: str = "/app/models/siformer/weights/siformer_bisindo_v2.pt"
    siformer_num_classes: int = 300              # BISINDO gloss vocabulary size
    siformer_sequence_len: int = 64              # Frames per inference window
    siformer_confidence_threshold: float = 0.90

    # ── Whisper ───────────────────────────────────────────────────
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = "large"
    whisper_fine_tuned_path: str | None = None   # Path to fine-tuned checkpoint
    whisper_language: str = "id"                 # Indonesian
    whisper_device: str = "cuda"

    # ── Inference Hardware ────────────────────────────────────────
    model_device: Literal["cuda", "cpu"] = "cuda"
    mediapipe_model_complexity: int = 1          # 0=lite, 1=full, 2=heavy

    # ── RLHF ──────────────────────────────────────────────────────
    rlhf_export_threshold: int = 500
    rlhf_edit_delta_threshold: float = 15.0
    rlhf_scheduler_interval_hours: int = 24

    # ── Socket.io (emitting to Node relay) ───────────────────────
    socket_namespace: str = "/ml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## 3. Data Strategy & Preprocessing Pipeline

### 3.1 Referenced Datasets

| Dataset | Source | Content | Usage |
|---|---|---|---|
| BISINDO Hand Sign Detection | `Rhiosutoyo/BISINDO-Hand-Sign-Detection-Dataset` (HuggingFace/GitHub) | Labeled static hand sign images for 26 BISINDO alphabets | Alphabet-level bootstrapping and MediaPipe calibration |
| BISINDO Roboflow Project | `BISINDO Project L92HB` (Roboflow) | Annotated image dataset with bounding boxes | Augmentation reference, hand detection pre-training |
| CNN-BISINDO Baseline | `ademaulana/CNN-BISINDO` (Hugging Face) | Pre-trained CNN for static BISINDO classification | Baseline accuracy benchmark; feature transfer |
| Mendeley BISINDO Dataset | Mendeley Data | Full video sequences of BISINDO words/phrases | **Primary training data** for Siformer temporal model |

### 3.2 Data Preprocessing Philosophy

The Siformer model is **skeleton-based**, not pixel-based. This means:

- Raw video frames → MediaPipe extracts `(x, y, z)` landmark coordinates
- Pixel data is **never stored** (privacy compliance)
- Each sample is a normalized matrix of shape `[T, J, 3]` where `T` = frames (sequence length), `J` = joints (landmarks), `3` = `(x, y, z)`

**Joint configuration (J = 75 total):**
- Left hand: 21 landmarks
- Right hand: 21 landmarks
- Pose (upper body): 25 landmarks
- Face (key points): 8 landmarks *(nose, eyes, mouth corners)*

### 3.3 `services/landmark_processor.py` — Extraction & Normalization

```python
# app/services/landmark_processor.py

import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional
import cv2


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 25   # Upper body only (indices 0-24 from full 33)
NUM_FACE_LANDMARKS = 8    # Key facial points only
NUM_JOINTS = (NUM_HAND_LANDMARKS * 2) + NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
FEATURE_DIM = 3            # (x, y, z)

# Pose landmark indices to keep (upper body)
UPPER_BODY_POSE_INDICES = list(range(25))

# Key facial landmark indices from MediaPipe Face Mesh (468 total → 8 key)
KEY_FACE_INDICES = [1, 4, 33, 263, 61, 291, 199, 0]  # nose tip, nose bridge, eyes, mouth


@dataclass
class FrameLandmarks:
    """Structured landmark output for a single frame."""
    left_hand: np.ndarray        # (21, 3) — zeros if not detected
    right_hand: np.ndarray       # (21, 3) — zeros if not detected
    pose: np.ndarray             # (25, 3)
    face: np.ndarray             # (8, 3)
    left_hand_visible: bool
    right_hand_visible: bool
    occlusion_score: float       # 0.0 (fully visible) → 1.0 (fully occluded)
    raw_vector: np.ndarray       # (75, 3) flattened — model input


class LandmarkProcessor:
    """
    Extracts and normalizes MediaPipe landmarks from video frames.

    Normalization strategy (signer-independence):
      1. Translate: anchor all coordinates to the nose tip (pose landmark 0)
      2. Scale: divide by the shoulder width (distance between landmarks 11 and 12)
         to make the representation scale-invariant across signers
      3. Clip: clamp extreme outliers to [-2.0, 2.0]
    """

    def __init__(self, model_complexity: int = 1):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """
        Process a single BGR frame and return normalized landmarks.

        Args:
            frame_bgr: OpenCV BGR frame, shape (H, W, 3)

        Returns:
            FrameLandmarks with normalized coordinates
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        left_hand, lh_visible = self._extract_hand(results.left_hand_landmarks)
        right_hand, rh_visible = self._extract_hand(results.right_hand_landmarks)
        pose = self._extract_pose(results.pose_landmarks)
        face = self._extract_face(results.face_landmarks)

        # Compute occlusion score before normalization
        occlusion_score = self._compute_occlusion_score(
            results, lh_visible, rh_visible
        )

        # Normalize entire coordinate space for signer-independence
        anchor, scale = self._compute_normalization_params(pose)
        left_hand = self._normalize(left_hand, anchor, scale)
        right_hand = self._normalize(right_hand, anchor, scale)
        pose = self._normalize(pose, anchor, scale)
        face = self._normalize(face, anchor, scale)

        raw_vector = np.concatenate([left_hand, right_hand, pose, face], axis=0)
        assert raw_vector.shape == (NUM_JOINTS, FEATURE_DIM), \
            f"Unexpected landmark shape: {raw_vector.shape}"

        return FrameLandmarks(
            left_hand=left_hand,
            right_hand=right_hand,
            pose=pose,
            face=face,
            left_hand_visible=lh_visible,
            right_hand_visible=rh_visible,
            occlusion_score=occlusion_score,
            raw_vector=raw_vector,
        )

    # ── Private Extraction Helpers ────────────────────────────────────────────

    def _extract_hand(
        self, landmarks
    ) -> tuple[np.ndarray, bool]:
        """Extract 21 hand landmarks. Returns zero array if not detected."""
        if landmarks is None:
            return np.zeros((NUM_HAND_LANDMARKS, FEATURE_DIM), dtype=np.float32), False

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32
        )
        return coords, True

    def _extract_pose(self, landmarks) -> np.ndarray:
        """Extract upper-body pose landmarks (first 25 of 33)."""
        if landmarks is None:
            return np.zeros((NUM_POSE_LANDMARKS, FEATURE_DIM), dtype=np.float32)

        all_pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32
        )
        return all_pose[UPPER_BODY_POSE_INDICES]

    def _extract_face(self, landmarks) -> np.ndarray:
        """Extract 8 key facial landmarks from MediaPipe Face Mesh."""
        if landmarks is None:
            return np.zeros((NUM_FACE_LANDMARKS, FEATURE_DIM), dtype=np.float32)

        all_face = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32
        )
        return all_face[KEY_FACE_INDICES]

    def _compute_normalization_params(
        self, pose: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Anchor = nose tip (pose[0]).
        Scale = shoulder width (Euclidean distance between pose[11] and pose[12]).
        Falls back to 1.0 if pose is all zeros (not detected).
        """
        anchor = pose[0].copy()  # Nose tip

        left_shoulder = pose[11]
        right_shoulder = pose[12]
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

        scale = shoulder_width if shoulder_width > 1e-6 else 1.0
        return anchor, scale

    def _normalize(
        self, coords: np.ndarray, anchor: np.ndarray, scale: float
    ) -> np.ndarray:
        """Translate to anchor, scale to shoulder width, clip outliers."""
        normalized = (coords - anchor) / scale
        return np.clip(normalized, -2.0, 2.0).astype(np.float32)

    def _compute_occlusion_score(
        self, results, lh_visible: bool, rh_visible: bool
    ) -> float:
        """
        Occlusion score: 0.0 = perfectly visible, 1.0 = fully occluded.

        Logic:
          - If both hands visible and pose detected → 0.0
          - If one hand missing → 0.4
          - If both hands missing → 0.8
          - If pose also missing → 1.0
        """
        pose_visible = results.pose_landmarks is not None
        hands_visible = int(lh_visible) + int(rh_visible)

        if hands_visible == 2 and pose_visible:
            return 0.0
        if hands_visible == 1 and pose_visible:
            return 0.4
        if hands_visible == 0 and pose_visible:
            return 0.8
        return 1.0

    def close(self):
        self.holistic.close()
```

### 3.4 `scripts/preprocess_dataset.py` — Offline Pipeline

```python
# scripts/preprocess_dataset.py
"""
Offline preprocessing: converts raw video files into normalized skeleton .npy sequences.
Run once before training:
    python scripts/preprocess_dataset.py \
        --input_dir data/raw/mendeley_bisindo \
        --output_dir data/processed \
        --seq_len 64 \
        --stride 32
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from app.services.landmark_processor import LandmarkProcessor, NUM_JOINTS, FEATURE_DIM


def extract_sequence(
    video_path: str,
    processor: LandmarkProcessor,
    seq_len: int = 64,
    stride: int = 32,
) -> list[np.ndarray]:
    """
    Extract overlapping fixed-length landmark sequences from a video file.

    Returns a list of arrays, each shape (seq_len, NUM_JOINTS, FEATURE_DIM).
    Sliding window with 'stride' overlap for data augmentation.
    """
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = processor.process_frame(frame)
        frames.append(landmarks.raw_vector)   # (75, 3)

    cap.release()

    if len(frames) < seq_len:
        # Pad with zero frames if video is shorter than sequence length
        pad = [np.zeros((NUM_JOINTS, FEATURE_DIM), dtype=np.float32)] * (
            seq_len - len(frames)
        )
        frames = frames + pad

    # Sliding window extraction
    sequences = []
    for start in range(0, len(frames) - seq_len + 1, stride):
        seq = np.stack(frames[start : start + seq_len], axis=0)  # (T, J, 3)
        sequences.append(seq)

    return sequences


def build_label_map(raw_dir: Path) -> dict[str, int]:
    """Build gloss → class index from directory names."""
    glosses = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    return {gloss: idx for idx, gloss in enumerate(glosses)}


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    seq_len: int = 64,
    stride: int = 32,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor = LandmarkProcessor(model_complexity=1)
    label_map = build_label_map(input_path)

    # Save label map
    with open(output_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"Found {len(label_map)} BISINDO gloss classes")

    all_sequences, all_labels = [], []

    for gloss, label_idx in tqdm(label_map.items(), desc="Processing glosses"):
        gloss_dir = input_path / gloss
        if not gloss_dir.exists():
            continue

        for video_file in gloss_dir.glob("*.mp4"):
            sequences = extract_sequence(
                str(video_file), processor, seq_len=seq_len, stride=stride
            )
            all_sequences.extend(sequences)
            all_labels.extend([label_idx] * len(sequences))

    X = np.stack(all_sequences, axis=0)   # (N, T, J, 3)
    y = np.array(all_labels, dtype=np.int64)  # (N,)

    print(f"Total samples: {len(X)} | Shape: {X.shape}")

    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)

    processor.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir, args.seq_len, args.stride)
```

### 3.5 Data Augmentation Strategy

To improve signer-independence and robustness, apply these augmentations **during training only** (not inference):

| Augmentation | Implementation | Purpose |
|---|---|---|
| **Temporal Flip** | Reverse the frame sequence order | Learn sign direction invariance |
| **Spatial Jitter** | Add Gaussian noise `N(0, 0.02)` to coordinates | Robustness to landmark jitter |
| **Speed Perturbation** | Resample sequence to ±20% of original speed via interpolation | Handle different signing speeds |
| **Mirror (Horizontal Flip)** | Negate x-axis, swap left/right hand arrays | Handle left-handed signers |
| **Random Frame Drop** | Drop 10% of frames randomly, re-pad | Robustness to dropped camera frames |
| **Coordinate Masking** | Zero-out one hand randomly (15% probability) | Train occlusion robustness |

```python
# Training augmentation applied in DataLoader collate_fn

import numpy as np
import torch
from scipy.interpolate import interp1d


def augment_sequence(
    seq: np.ndarray,          # (T, J, 3)
    p_flip: float = 0.5,
    p_jitter: float = 0.7,
    p_speed: float = 0.4,
    p_mirror: float = 0.3,
    p_mask: float = 0.15,
) -> np.ndarray:
    T, J, C = seq.shape

    # Temporal flip
    if np.random.random() < p_flip:
        seq = seq[::-1].copy()

    # Spatial jitter
    if np.random.random() < p_jitter:
        noise = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
        seq = np.clip(seq + noise, -2.0, 2.0)

    # Speed perturbation (±20%)
    if np.random.random() < p_speed:
        speed_factor = np.random.uniform(0.8, 1.2)
        new_T = max(int(T * speed_factor), 1)
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, new_T)
        resampled = np.zeros((new_T, J, C), dtype=np.float32)
        for j in range(J):
            for c in range(C):
                f = interp1d(old_indices, seq[:, j, c], kind="linear")
                resampled[:, j, c] = f(new_indices)
        # Pad or truncate back to T
        if new_T < T:
            pad = np.zeros((T - new_T, J, C), dtype=np.float32)
            seq = np.concatenate([resampled, pad], axis=0)
        else:
            seq = resampled[:T]

    # Mirror (negate x, swap left/right hands)
    # Left hand: indices 0-20, Right hand: indices 21-41
    if np.random.random() < p_mirror:
        seq[:, :, 0] *= -1   # Negate x-axis
        left = seq[:, :21, :].copy()
        right = seq[:, 21:42, :].copy()
        seq[:, :21, :] = right
        seq[:, 21:42, :] = left

    # Random hand masking (occlusion simulation)
    if np.random.random() < p_mask:
        hand_to_mask = np.random.choice([0, 1])  # 0=left, 1=right
        start_idx = 0 if hand_to_mask == 0 else 21
        seq[:, start_idx : start_idx + 21, :] = 0.0

    return seq
```

---

## 4. Siformer: Spatial-Temporal Transformer

### 4.1 Architecture Overview

Siformer processes skeleton sequences as follows:

```
Input: (B, T, J, 3)  →  B=batch, T=64 frames, J=75 joints, 3=(x,y,z)
         │
         ▼
┌─────────────────────┐
│  Joint Embedding    │  Linear projection: (J, 3) → (J, d_model)
│  (per-frame)        │  + Learnable joint positional encoding
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Spatial Encoder    │  Multi-head self-attention across J joints
│  (N_s layers)       │  Captures hand shape context per frame
│                     │  Input/Output: (B*T, J, d_model)
└─────────┬───────────┘
          │
          ▼ Reshape: (B, T, J*d_model) → mean-pool over J → (B, T, d_model)
          │
          ▼
┌─────────────────────┐
│  Temporal Encoder   │  Multi-head self-attention across T frames
│  (N_t layers)       │  + Sinusoidal temporal positional encoding
│                     │  Captures motion dynamics over time
│                     │  Input/Output: (B, T, d_model)
└─────────┬───────────┘
          │ CLS token pooling → (B, d_model)
          │
          ▼
┌─────────────────────┐
│  Classification     │  Linear(d_model → num_classes) + Softmax
│  Head               │
└─────────────────────┘
Output: (B, num_classes) logits
```

### 4.2 `models/siformer/architecture.py`

```python
# app/models/siformer/architecture.py

import math
import torch
import torch.nn as nn
from einops import rearrange


# ── Positional Encodings ──────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal axis."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnableJointEncoding(nn.Module):
    """Learnable positional encoding for the joint (spatial) axis."""

    def __init__(self, num_joints: int, d_model: int):
        super().__init__()
        self.encoding = nn.Embedding(num_joints, d_model)
        self.register_buffer(
            "joint_ids", torch.arange(num_joints).unsqueeze(0)  # (1, J)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_T, J, d_model)
        return x + self.encoding(self.joint_ids)


# ── Transformer Building Blocks ───────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with pre-norm (more stable training).
    Pre-norm: LayerNorm before attention and FFN sublayers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


# ── Siformer ─────────────────────────────────────────────────────────────────

class Siformer(nn.Module):
    """
    Spatial-Temporal Transformer for Skeleton-based Sign Language Recognition.

    Two-stage encoding:
      1. Spatial Encoder: Models joint relationships within each frame
      2. Temporal Encoder: Models motion dynamics across frames

    Args:
        num_joints:    Number of body landmarks (default: 75)
        in_channels:   Input feature dim per joint (default: 3 for x,y,z)
        d_model:       Hidden dimension (default: 256)
        n_spatial:     Number of spatial encoder layers (default: 4)
        n_temporal:    Number of temporal encoder layers (default: 4)
        n_heads:       Number of attention heads (default: 8)
        ffn_dim:       Feed-forward hidden dim (default: 512)
        num_classes:   Output vocabulary size (default: 300 for BISINDO)
        seq_len:       Temporal sequence length (default: 64)
        dropout:       Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        num_joints: int = 75,
        in_channels: int = 3,
        d_model: int = 256,
        n_spatial: int = 4,
        n_temporal: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 512,
        num_classes: int = 300,
        seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_joints = num_joints
        self.seq_len = seq_len

        # ── Input Projection ────────────────────────────────────────────────
        # Projects each joint's (x,y,z) features to d_model dimensions
        self.joint_projection = nn.Linear(in_channels, d_model)

        # ── Spatial Encoding ────────────────────────────────────────────────
        self.joint_pos_enc = LearnableJointEncoding(num_joints, d_model)
        self.spatial_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_spatial)
        ])
        self.spatial_norm = nn.LayerNorm(d_model)

        # Aggregate joint features per frame: mean pool over J dim
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)  # (J, d_model) → (d_model,)

        # ── Temporal Encoding ────────────────────────────────────────────────
        # CLS token prepended to temporal sequence for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=seq_len + 1, dropout=dropout
        )
        self.temporal_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_temporal)
        ])
        self.temporal_norm = nn.LayerNorm(d_model)

        # ── Classification Head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, J, 3) — batch of skeleton sequences
            padding_mask: (B, T) — True where frame is padded (zero frame)

        Returns:
            logits: (B, num_classes)
            attn_weights: (B, T+1, T+1) — temporal attention for visualization
        """
        B, T, J, C = x.shape

        # ── Stage 1: Spatial Encoding ───────────────────────────────────────
        # Reshape to process all frames independently
        x = rearrange(x, "b t j c -> (b t) j c")     # (B*T, J, 3)

        # Project joints to d_model
        x = self.joint_projection(x)                   # (B*T, J, d_model)
        x = self.joint_pos_enc(x)                      # + learnable joint pos

        # Apply spatial transformer layers
        for layer in self.spatial_encoder:
            x = layer(x)
        x = self.spatial_norm(x)                       # (B*T, J, d_model)

        # Pool across joints → single frame representation
        x = x.permute(0, 2, 1)                        # (B*T, d_model, J)
        x = self.spatial_pool(x).squeeze(-1)           # (B*T, d_model)

        # Reshape back to temporal dimension
        x = rearrange(x, "(b t) d -> b t d", b=B, t=T)  # (B, T, d_model)

        # ── Stage 2: Temporal Encoding ──────────────────────────────────────
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                 # (B, T+1, d_model)

        # Add temporal positional encoding
        x = self.temporal_pos_enc(x)

        # Extend padding mask for CLS token (never masked)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Apply temporal transformer layers
        for layer in self.temporal_encoder:
            x = layer(x, src_key_padding_mask=padding_mask)
        x = self.temporal_norm(x)

        # Extract CLS token as global sequence representation
        cls_output = x[:, 0, :]                        # (B, d_model)
        attn_weights = x                               # Return for visualization

        # ── Classification ───────────────────────────────────────────────────
        logits = self.classifier(cls_output)            # (B, num_classes)

        return logits, attn_weights

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[int, float, torch.Tensor]:
        """
        Single-sample inference.

        Returns:
            class_idx: Predicted class index
            confidence: Softmax probability of top prediction (0.0–1.0)
            probs: Full probability distribution (num_classes,)
        """
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)   # Add batch dim: (T, J, 3) → (1, T, J, 3)

        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

        return class_idx, confidence, probs
```

### 4.3 `models/siformer/loader.py`

```python
# app/models/siformer/loader.py

import torch
import json
from pathlib import Path
from functools import lru_cache

from app.models.siformer.architecture import Siformer
from app.core.config import get_settings
import structlog

logger = structlog.get_logger()


class SiformerLoader:
    """Thread-safe singleton loader for the Siformer model."""

    _instance: Siformer | None = None
    _label_map: dict[int, str] | None = None

    @classmethod
    def get_model(cls) -> Siformer:
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def get_label_map(cls) -> dict[int, str]:
        """Returns int → gloss string mapping."""
        if cls._label_map is None:
            settings = get_settings()
            label_path = Path(settings.siformer_model_path).parent / "bisindo_labels.json"
            with open(label_path) as f:
                gloss_to_idx: dict[str, int] = json.load(f)
            cls._label_map = {v: k for k, v in gloss_to_idx.items()}
        return cls._label_map

    @classmethod
    def _load(cls) -> Siformer:
        settings = get_settings()
        device = torch.device(settings.model_device)

        logger.info("Loading Siformer", path=settings.siformer_model_path, device=str(device))

        checkpoint = torch.load(settings.siformer_model_path, map_location=device)

        # Support both raw state_dict and checkpoint dicts
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            model_config = checkpoint.get("model_config", {})
        else:
            state_dict = checkpoint
            model_config = {}

        model = Siformer(
            num_classes=settings.siformer_num_classes,
            seq_len=settings.siformer_sequence_len,
            **model_config,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        # Warm up with a dummy forward pass
        dummy = torch.zeros(1, settings.siformer_sequence_len, 75, 3, device=device)
        with torch.no_grad():
            model(dummy)

        logger.info("Siformer loaded and warmed up", num_classes=settings.siformer_num_classes)
        return model
```

### 4.4 Training Script

```python
# scripts/train_siformer.py

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from app.models.siformer.architecture import Siformer
from app.services.landmark_processor import NUM_JOINTS, FEATURE_DIM


class BISINDODataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()   # (N, T, J, 3)
        self.y = torch.from_numpy(y).long()     # (N,)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return x, self.y[idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Import augmentation from preprocessing module
        from scripts.preprocess_dataset import augment_sequence
        x_np = augment_sequence(x.numpy())
        return torch.from_numpy(x_np).float()


def train(
    data_dir: str = "data/processed",
    output_dir: str = "app/models/siformer/weights",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 256,
    n_heads: int = 8,
    n_spatial: int = 4,
    n_temporal: int = 4,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    X = np.load(data_path / "X.npy")           # (N, T, J, 3)
    y = np.load(data_path / "y.npy")           # (N,)
    num_classes = int(y.max()) + 1

    print(f"Dataset: {len(X)} samples | {num_classes} classes")

    # Train/Val split (80/20)
    dataset = BISINDODataset(X, y, augment=False)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_set.dataset.augment = True   # Enable augmentation for training split

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Siformer(
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_spatial=n_spatial,
        n_temporal=n_temporal,
        seq_len=X.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "model_config": {
                        "num_joints": 75,
                        "in_channels": 3,
                        "d_model": d_model,
                        "n_spatial": n_spatial,
                        "n_temporal": n_temporal,
                        "n_heads": n_heads,
                        "ffn_dim": d_model * 2,
                        "num_classes": num_classes,
                        "seq_len": X.shape[1],
                    },
                },
                output_path / "siformer_bisindo_v2.pt",
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
```

---

## 5. Voice Engine: Whisper Integration

### 5.1 `models/whisper/loader.py`

```python
# app/models/whisper/loader.py

import whisper
import torch
from pathlib import Path
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()


class WhisperLoader:
    """Singleton loader for Whisper model (base or fine-tuned)."""

    _model: whisper.Whisper | None = None

    @classmethod
    def get_model(cls) -> whisper.Whisper:
        if cls._model is None:
            cls._model = cls._load()
        return cls._model

    @classmethod
    def _load(cls) -> whisper.Whisper:
        settings = get_settings()
        device = settings.whisper_device

        logger.info("Loading Whisper", size=settings.whisper_model_size, device=device)

        if settings.whisper_fine_tuned_path and Path(settings.whisper_fine_tuned_path).exists():
            # Load fine-tuned checkpoint
            logger.info("Using fine-tuned Whisper", path=settings.whisper_fine_tuned_path)
            model = whisper.load_model(settings.whisper_model_size, device=device)
            checkpoint = torch.load(settings.whisper_fine_tuned_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Load base Whisper model
            model = whisper.load_model(settings.whisper_model_size, device=device)

        model.eval()
        logger.info("Whisper loaded successfully")
        return model
```

### 5.2 `services/voice_inference.py`

```python
# app/services/voice_inference.py

import io
import time
import uuid
import numpy as np
import whisper
import torch
from typing import AsyncGenerator

import structlog

from app.models.whisper.loader import WhisperLoader
from app.core.config import get_settings
from app.schemas.voice import VoiceInferenceResult, AudioChunk

logger = structlog.get_logger()


class VoiceInferenceService:
    """
    Handles speech-to-text inference using Whisper.

    Supports:
      - Single-chunk transcription (short utterances)
      - Streaming mode (chunked audio accumulation)
    """

    SAMPLE_RATE = 16_000   # Whisper always expects 16kHz mono

    def __init__(self):
        self.model = WhisperLoader.get_model()
        self.settings = get_settings()

    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        session_token: str,
        inference_id: str | None = None,
    ) -> VoiceInferenceResult:
        """
        Transcribe a single audio chunk (WAV bytes at 16kHz mono).

        Args:
            audio_bytes: Raw WAV audio bytes
            session_token: Conversation session identifier
            inference_id: Optional ID to link streaming events

        Returns:
            VoiceInferenceResult with transcription and confidence
        """
        if inference_id is None:
            inference_id = str(uuid.uuid4())

        start_time = time.monotonic()

        # Decode audio to numpy float32 array
        audio_array = self._bytes_to_array(audio_bytes)

        # Run Whisper inference
        result = self.model.transcribe(
            audio_array,
            language=self.settings.whisper_language,   # "id" for Indonesian
            task="transcribe",
            fp16=torch.cuda.is_available(),
            without_timestamps=True,
            condition_on_previous_text=False,           # Better for short utterances
        )

        latency_ms = int((time.monotonic() - start_time) * 1000)
        text = result["text"].strip()
        confidence = self._extract_confidence(result)

        logger.info(
            "Voice transcription complete",
            session=session_token,
            text_preview=text[:50],
            confidence=confidence,
            latency_ms=latency_ms,
        )

        return VoiceInferenceResult(
            inference_id=inference_id,
            session_token=session_token,
            raw_prediction_text=text,
            confidence_score=confidence,
            model_version=f"whisper-{self.settings.whisper_model_size}",
            inference_latency_ms=latency_ms,
        )

    async def transcribe_streaming(
        self,
        audio_chunks: list[bytes],
        session_token: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming transcription: yields partial words as they are decoded.

        Whisper is not natively token-streaming, so we implement pseudo-streaming:
        - Decode in segments using Whisper's segment output
        - Yield each segment's text as it completes
        - This gives ~1-3s perceived latency reduction vs waiting for full audio

        Args:
            audio_chunks: List of audio byte chunks (concatenated)
            session_token: Conversation session identifier

        Yields:
            Partial transcription text segments
        """
        # Concatenate all chunks into one buffer
        combined = b"".join(audio_chunks)
        audio_array = self._bytes_to_array(combined)

        # Use Whisper's segment-level output for pseudo-streaming
        result = self.model.transcribe(
            audio_array,
            language=self.settings.whisper_language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            without_timestamps=False,   # Need timestamps for segment streaming
        )

        accumulated = ""
        for segment in result.get("segments", []):
            segment_text = segment["text"].strip()
            if segment_text:
                accumulated += " " + segment_text
                yield accumulated.strip()

    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to float32 numpy array at 16kHz."""
        # Use whisper's built-in loader which handles resampling
        with io.BytesIO(audio_bytes) as buf:
            audio = whisper.load_audio(buf)
        return audio   # Already float32, normalized to [-1, 1]

    def _extract_confidence(self, whisper_result: dict) -> float:
        """
        Extract a confidence proxy from Whisper's segment avg_logprob.

        Whisper returns avg_logprob per segment (negative log-prob).
        Map to [0, 1] range:
          logprob = 0.0  → confidence = 1.0 (perfect)
          logprob = -1.0 → confidence = 0.37
          logprob < -1.5 → confidence < 0.22 (likely poor transcription)
        """
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.5   # Default uncertainty

        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
        # Sigmoid-like mapping: e^logprob
        confidence = float(np.exp(np.clip(avg_logprob, -5.0, 0.0)))
        return round(confidence, 4)
```

---

## 6. FastAPI Streaming Architecture

### 6.1 `main.py` — Application Factory

```python
# app/main.py

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1 import sign, voice, rlhf
from app.core.config import get_settings
from app.core.events import startup_handler, shutdown_handler
from app.core.logging import configure_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    await startup_handler()
    yield
    await shutdown_handler()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="SLOP ML Service",
        description="Sign Language Recognition & Voice Transcription API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
    )

    # CORS — restrict to Node.js server in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.node_server_url],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # Register routers
    app.include_router(sign.router, prefix="/v1/sign", tags=["Sign Language"])
    app.include_router(voice.router, prefix="/v1/voice", tags=["Voice"])
    app.include_router(rlhf.router, prefix="/v1/rlhf", tags=["RLHF"])

    @app.get("/v1/health")
    async def health():
        return {
            "status": "healthy",
            "service": settings.service_name,
            "environment": settings.environment,
        }

    return app


app = create_app()
```

### 6.2 `core/events.py` — Model Preloading

```python
# app/core/events.py

import structlog
from app.models.siformer.loader import SiformerLoader
from app.models.whisper.loader import WhisperLoader
from app.services.rlhf_scheduler import RLHFScheduler

logger = structlog.get_logger()
_scheduler: RLHFScheduler | None = None


async def startup_handler() -> None:
    """Pre-load all models on startup to avoid cold-start latency during requests."""
    logger.info("Starting SLOP ML Service — preloading models")

    # Block until both models are loaded
    SiformerLoader.get_model()
    SiformerLoader.get_label_map()
    WhisperLoader.get_model()

    # Start background RLHF scheduler
    global _scheduler
    _scheduler = RLHFScheduler()
    await _scheduler.start()

    logger.info("All models loaded. Service ready.")


async def shutdown_handler() -> None:
    """Graceful shutdown: stop background tasks."""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
    logger.info("ML Service shutdown complete")
```

### 6.3 `api/v1/sign.py` — WebSocket Streaming Endpoint

```python
# app/api/v1/sign.py

import asyncio
import base64
import json
import time
import uuid
from typing import Any

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.sign import (
    SignInferenceRequest,
    SignInferenceResponse,
    BoundingBox,
)
from app.services.sign_inference import SignInferenceService
from app.services.socketio_emitter import SocketIOEmitter
from app.core.config import get_settings

logger = structlog.get_logger()
router = APIRouter()

# Module-level singleton services (initialized once at startup)
_sign_service: SignInferenceService | None = None
_emitter: SocketIOEmitter | None = None


def get_sign_service() -> SignInferenceService:
    global _sign_service
    if _sign_service is None:
        _sign_service = SignInferenceService()
    return _sign_service


def get_emitter() -> SocketIOEmitter:
    global _emitter
    if _emitter is None:
        _emitter = SocketIOEmitter()
    return _emitter


# ── REST: Single Frame Inference ──────────────────────────────────────────────

@router.post("/infer", response_model=SignInferenceResponse)
async def infer_single_frame(request: SignInferenceRequest) -> SignInferenceResponse:
    """
    Single-frame sign inference endpoint.
    Accepts a base64-encoded frame and returns prediction + confidence.
    Used for testing and single-shot recognition.
    """
    service = get_sign_service()
    try:
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        result = await service.infer_frame(frame, request.session_token)
        return result

    except Exception as e:
        logger.error("Single frame inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket: Real-Time Streaming ───────────────────────────────────────────

@router.websocket("/stream")
async def sign_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time sign language inference.

    Protocol:
      CLIENT → SERVER: JSON handshake with session_token, then binary frames
      SERVER → CLIENT: JSON events (inference_started, text_streamed, inference_complete)

    Frame format (client sends):
      {
        "type": "frame",
        "session_token": "abc123",
        "frame_base64": "<base64 JPEG>",
        "timestamp": 1714000000.0
      }

    Or raw binary: JPEG bytes directly (after handshake)
    """
    await websocket.accept()
    settings = get_settings()
    service = get_sign_service()
    emitter = get_emitter()

    session_token: str | None = None
    inference_id: str | None = None
    frame_buffer: list[np.ndarray] = []
    last_inference_time = 0.0
    INFERENCE_INTERVAL_S = 2.0   # Trigger full inference every 2 seconds of sign activity

    logger.info("WebSocket connection opened")

    try:
        # ── Handshake: first message must identify the session ──────────────
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        handshake = json.loads(raw)
        session_token = handshake.get("session_token")

        if not session_token:
            await websocket.send_json({"error": "session_token required in handshake"})
            await websocket.close(code=1008)
            return

        inference_id = str(uuid.uuid4())
        logger.info("WS session identified", session=session_token)

        # Emit inference_started to Node.js → relay to client
        await emitter.emit_inference_started(
            session_token=session_token,
            inference_id=inference_id,
            modality="SIGN",
        )
        await websocket.send_json({
            "event": "inference_started",
            "inference_id": inference_id,
        })

        # ── Frame receive loop ───────────────────────────────────────────────
        while True:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=30.0  # 30s idle timeout
            )

            # Handle binary frame (raw JPEG bytes)
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
            # Handle JSON frame with base64
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "frame":
                    frame_bytes = base64.b64decode(data["frame_base64"])
                elif data.get("type") == "end_signing":
                    # Client signals end of sign gesture — run final inference
                    break
                else:
                    continue
            else:
                continue

            # Decode frame
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Process landmarks for this frame
            frame_result = await service.process_single_frame(frame)

            # Stream bounding box + confidence update to client
            text_streamed_payload = {
                "event": "text_streamed",
                "inference_id": inference_id,
                "partial_text": "",    # Populated after buffer inference
                "confidence_score": frame_result.frame_confidence,
                "occlusion_detected": frame_result.occlusion_detected,
                "bounding_box": frame_result.bounding_box.model_dump()
                if frame_result.bounding_box else None,
            }
            await websocket.send_json(text_streamed_payload)

            # Accumulate frames into buffer
            frame_buffer.append(frame_result.landmarks.raw_vector)

            # Run full Siformer inference every INFERENCE_INTERVAL_S
            now = time.monotonic()
            if (
                len(frame_buffer) >= settings.siformer_sequence_len
                and now - last_inference_time >= INFERENCE_INTERVAL_S
            ):
                inference_result = await service.infer_buffer(
                    frame_buffer=frame_buffer[-settings.siformer_sequence_len :],
                    session_token=session_token,
                    inference_id=inference_id,
                )
                last_inference_time = now

                # Stream partial prediction
                streamed_payload = {
                    **text_streamed_payload,
                    "partial_text": inference_result.raw_prediction_text,
                    "confidence_score": inference_result.confidence_score,
                }
                await websocket.send_json(streamed_payload)
                await emitter.emit_text_streamed(
                    session_token=session_token,
                    inference_id=inference_id,
                    partial_text=inference_result.raw_prediction_text,
                    confidence_score=inference_result.confidence_score,
                    occlusion_detected=frame_result.occlusion_detected,
                    bounding_box=frame_result.bounding_box,
                )

        # ── Final inference on remaining buffer ──────────────────────────────
        if len(frame_buffer) >= 8:   # Minimum frames for meaningful prediction
            final_result = await service.infer_buffer(
                frame_buffer=frame_buffer[-settings.siformer_sequence_len :],
                session_token=session_token,
                inference_id=inference_id,
            )

            complete_payload = {
                "event": "inference_complete",
                "inference_id": inference_id,
                "raw_prediction_text": final_result.raw_prediction_text,
                "final_confidence_score": final_result.confidence_score,
                "model_version": final_result.model_version,
                "inference_latency_ms": final_result.inference_latency_ms,
                "occlusion_detected": final_result.occlusion_detected,
            }
            await websocket.send_json(complete_payload)
            await emitter.emit_inference_complete(
                session_token=session_token,
                payload=complete_payload,
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", session=session_token)
    except asyncio.TimeoutError:
        logger.warning("WebSocket idle timeout", session=session_token)
        await websocket.close(code=1001)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), session=session_token)
        await websocket.close(code=1011)
```

### 6.4 `api/v1/voice.py` — Audio Transcription Endpoint

```python
# app/api/v1/voice.py

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import json

import structlog

from app.services.voice_inference import VoiceInferenceService
from app.services.socketio_emitter import SocketIOEmitter
from app.schemas.voice import VoiceInferenceResult

logger = structlog.get_logger()
router = APIRouter()

_voice_service: VoiceInferenceService | None = None
_emitter: SocketIOEmitter | None = None


def get_voice_service() -> VoiceInferenceService:
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceInferenceService()
    return _voice_service


@router.post("/transcribe", response_model=VoiceInferenceResult)
async def transcribe_audio(
    audio: UploadFile = File(..., description="WAV audio file at 16kHz mono"),
    session_token: str = Form(...),
    inference_id: str | None = Form(None),
) -> VoiceInferenceResult:
    """
    Transcribe a single WAV audio chunk.
    Audio must be 16kHz mono WAV.
    Returns transcription with confidence score.
    """
    if not audio.content_type in ("audio/wav", "audio/wave", "audio/x-wav", "audio/webm"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {audio.content_type}. Use WAV or WebM.",
        )

    service = get_voice_service()
    emitter = get_emitter() or SocketIOEmitter()

    audio_bytes = await audio.read()
    inf_id = inference_id or str(uuid.uuid4())

    # Emit inference_started to Node.js
    await emitter.emit_inference_started(
        session_token=session_token,
        inference_id=inf_id,
        modality="VOICE",
    )

    result = await service.transcribe_chunk(
        audio_bytes=audio_bytes,
        session_token=session_token,
        inference_id=inf_id,
    )

    # Emit inference_complete to Node.js
    await emitter.emit_inference_complete(
        session_token=session_token,
        payload={
            "event": "inference_complete",
            "inference_id": inf_id,
            "raw_prediction_text": result.raw_prediction_text,
            "final_confidence_score": result.confidence_score,
            "model_version": result.model_version,
            "inference_latency_ms": result.inference_latency_ms,
            "occlusion_detected": False,
        },
    )

    return result


@router.post("/transcribe/stream")
async def transcribe_audio_streaming(
    audio: UploadFile = File(...),
    session_token: str = Form(...),
):
    """
    Pseudo-streaming transcription.
    Returns Server-Sent Events (SSE) with partial transcriptions as Whisper
    processes each segment.
    """
    service = get_voice_service()
    audio_bytes = await audio.read()

    async def event_generator():
        async for partial_text in service.transcribe_streaming(
            audio_chunks=[audio_bytes],
            session_token=session_token,
        ):
            data = json.dumps({"partial_text": partial_text})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

## 7. Confidence Score & Occlusion Detection

### 7.1 `services/sign_inference.py` — Full Orchestration

```python
# app/services/sign_inference.py

import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import structlog

from app.models.siformer.loader import SiformerLoader
from app.services.landmark_processor import LandmarkProcessor, FrameLandmarks, NUM_JOINTS
from app.services.occlusion_detector import OcclusionDetector
from app.schemas.sign import SignInferenceResponse, FrameInferenceResult, BoundingBox
from app.core.config import get_settings

logger = structlog.get_logger()


class SignInferenceService:
    """
    Orchestrates the full SLR pipeline:
      1. LandmarkProcessor: MediaPipe skeleton extraction
      2. OcclusionDetector: Visibility + bounding box computation
      3. Siformer: Temporal sequence classification
      4. Confidence aggregation
    """

    def __init__(self):
        self.settings = get_settings()
        self.model = SiformerLoader.get_model()
        self.label_map = SiformerLoader.get_label_map()
        self.landmark_processor = LandmarkProcessor(
            model_complexity=self.settings.mediapipe_model_complexity
        )
        self.occlusion_detector = OcclusionDetector()
        self.device = torch.device(self.settings.model_device)

    async def process_single_frame(self, frame_bgr: np.ndarray) -> FrameInferenceResult:
        """
        Process a single frame: extract landmarks, compute occlusion + bounding box.
        Does NOT run Siformer — that requires a full buffer.
        Used for streaming per-frame feedback (bounding box color, ghost text update).
        """
        landmarks = self.landmark_processor.process_frame(frame_bgr)
        bounding_box = self.occlusion_detector.compute_bounding_box(
            landmarks, frame_bgr.shape
        )
        frame_confidence = self._compute_frame_confidence(landmarks)

        return FrameInferenceResult(
            landmarks=landmarks,
            bounding_box=bounding_box,
            frame_confidence=frame_confidence,
            occlusion_detected=landmarks.occlusion_score > 0.3,
        )

    async def infer_buffer(
        self,
        frame_buffer: list[np.ndarray],   # List of (75, 3) landmark arrays
        session_token: str,
        inference_id: str | None = None,
    ) -> SignInferenceResponse:
        """
        Run full Siformer inference on a buffer of skeleton frames.

        Args:
            frame_buffer: List of (NUM_JOINTS, 3) normalized landmark arrays
            session_token: Conversation session ID
            inference_id: Optional ID for event linking

        Returns:
            SignInferenceResponse with prediction, confidence, latency
        """
        settings = self.settings
        if inference_id is None:
            inference_id = str(uuid.uuid4())

        seq_len = settings.siformer_sequence_len

        # Pad or truncate buffer to fixed sequence length
        padded = self._pad_sequence(frame_buffer, seq_len)

        # Convert to tensor: (1, T, J, 3)
        x = torch.from_numpy(padded).float().unsqueeze(0).to(self.device)

        # Create padding mask: True where frame was zero-padded
        is_padded = len(frame_buffer) < seq_len
        padding_mask = None
        if is_padded:
            mask = [False] * len(frame_buffer) + [True] * (seq_len - len(frame_buffer))
            padding_mask = torch.tensor([mask], dtype=torch.bool, device=self.device)

        start_time = time.monotonic()

        with torch.no_grad():
            class_idx, confidence, probs = self.model.predict(x)

        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Map class index to BISINDO gloss label
        gloss = self.label_map.get(class_idx, f"[UNK_{class_idx}]")

        # Aggregate confidence: weighted by top-3 entropy
        aggregated_confidence = self._aggregate_confidence(probs, top_k=3)

        logger.info(
            "Siformer inference complete",
            session=session_token,
            gloss=gloss,
            confidence=aggregated_confidence,
            latency_ms=latency_ms,
        )

        return SignInferenceResponse(
            inference_id=inference_id,
            session_token=session_token,
            raw_prediction_text=gloss,
            confidence_score=aggregated_confidence,
            class_idx=class_idx,
            model_version="siformer-bisindo-v2",
            inference_latency_ms=latency_ms,
            occlusion_detected=False,   # Populated by caller from frame-level result
        )

    def _compute_frame_confidence(self, landmarks: FrameLandmarks) -> float:
        """
        Per-frame confidence heuristic based on MediaPipe visibility scores.

        Higher = more reliable skeleton, lower = poor detection quality.
        Returns a value in [0.0, 1.0].
        """
        visibility_score = 1.0 - landmarks.occlusion_score

        # Penalize if zero-vectors detected (landmarks not found)
        left_nonzero = np.any(landmarks.left_hand != 0)
        right_nonzero = np.any(landmarks.right_hand != 0)
        hand_penalty = 0.0
        if not left_nonzero and not right_nonzero:
            hand_penalty = 0.4
        elif not left_nonzero or not right_nonzero:
            hand_penalty = 0.15

        return max(0.0, visibility_score - hand_penalty)

    def _aggregate_confidence(
        self, probs: torch.Tensor, top_k: int = 3
    ) -> float:
        """
        Aggregate confidence using top-k probability mass and entropy.

        High entropy (uniform distribution) → low confidence.
        Low entropy (peaked distribution) → high confidence.

        Returns value in [0.0, 1.0].
        """
        top_probs, _ = torch.topk(probs, k=min(top_k, len(probs)))
        top_mass = float(top_probs.sum().item())

        # Shannon entropy of full distribution (normalized to [0, 1])
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-9)).item())
        max_entropy = float(torch.log(torch.tensor(len(probs), dtype=torch.float)).item())
        normalized_entropy = entropy / (max_entropy + 1e-9)

        # Confidence = top-k mass weighted down by entropy
        confidence = top_mass * (1.0 - 0.3 * normalized_entropy)
        return round(float(np.clip(confidence, 0.0, 1.0)), 4)

    def _pad_sequence(
        self, frame_buffer: list[np.ndarray], seq_len: int
    ) -> np.ndarray:
        """Pad or truncate to fixed sequence length (T, J, 3)."""
        if len(frame_buffer) >= seq_len:
            return np.stack(frame_buffer[-seq_len:], axis=0)

        pad_frames = seq_len - len(frame_buffer)
        zero_pad = [np.zeros((NUM_JOINTS, 3), dtype=np.float32)] * pad_frames
        return np.stack(frame_buffer + zero_pad, axis=0)
```

### 7.2 `services/occlusion_detector.py`

```python
# app/services/occlusion_detector.py

import numpy as np
from app.services.landmark_processor import FrameLandmarks
from app.schemas.sign import BoundingBox


class OcclusionDetector:
    """
    Computes bounding box coordinates and occlusion state from MediaPipe landmarks.

    Bounding box is derived from visible hand + upper-body landmarks.
    Coordinates are returned as normalized [0, 1] values relative to frame size.
    """

    # Minimum hand landmark visibility threshold
    VISIBILITY_THRESHOLD = 0.5

    # Frame edge proximity threshold (landmark within this % of edge = near-occluded)
    EDGE_PROXIMITY = 0.05

    def compute_bounding_box(
        self,
        landmarks: FrameLandmarks,
        frame_shape: tuple[int, int, int],   # (H, W, C)
    ) -> BoundingBox | None:
        """
        Compute a bounding box around visible hand landmarks.

        Returns None if no hands are detected.
        Returns BoundingBox with:
          - Normalized (x, y, width, height) in [0, 1] range
          - is_occluded: True if hands are near frame edges
          - occlusion_score: float [0, 1]
        """
        H, W, _ = frame_shape

        visible_points: list[tuple[float, float]] = []

        # Collect visible landmark (x, y) from hands and pose
        if landmarks.left_hand_visible:
            for point in landmarks.left_hand:
                # Denormalize from landmark processor's [-2, 2] back to approx [0, 1]
                # Note: landmarks are relative to nose anchor; re-project to screen space
                visible_points.append((point[0], point[1]))

        if landmarks.right_hand_visible:
            for point in landmarks.right_hand:
                visible_points.append((point[0], point[1]))

        if not visible_points:
            return None

        xs = [p[0] for p in visible_points]
        ys = [p[1] for p in visible_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add padding around bounding box (10%)
        pad_x = (x_max - x_min) * 0.10
        pad_y = (y_max - y_min) * 0.10

        # Clamp to [0, 1]
        x = float(np.clip(x_min - pad_x, 0.0, 1.0))
        y = float(np.clip(y_min - pad_y, 0.0, 1.0))
        w = float(np.clip((x_max - x_min) + 2 * pad_x, 0.0, 1.0))
        h = float(np.clip((y_max - y_min) + 2 * pad_y, 0.0, 1.0))

        # Occlusion: check if bounding box clips frame edges
        is_occluded = (
            x < self.EDGE_PROXIMITY
            or y < self.EDGE_PROXIMITY
            or (x + w) > (1.0 - self.EDGE_PROXIMITY)
            or (y + h) > (1.0 - self.EDGE_PROXIMITY)
        )

        return BoundingBox(
            x=x,
            y=y,
            width=w,
            height=h,
            is_occluded=is_occluded,
            occlusion_score=landmarks.occlusion_score,
        )

    @staticmethod
    def get_ui_state(
        confidence_score: float,
        occlusion_detected: bool,
    ) -> dict:
        """
        Map confidence + occlusion to UI bounding box visual state.
        Consumed by the front-end to set box color.

        Returns:
          {
            "color": "#22C55E",
            "animated": false,
            "dash": false,
            "state": "HIGH_CONFIDENCE"
          }
        """
        if occlusion_detected:
            return {
                "color": "#F59E0B",
                "animated": False,
                "dash": True,
                "state": "OCCLUSION",
            }
        if confidence_score > 0.90:
            return {
                "color": "#22C55E",
                "animated": False,
                "dash": False,
                "state": "HIGH_CONFIDENCE",
            }
        return {
            "color": "#EF4444",
            "animated": True,
            "dash": False,
            "state": "LOW_CONFIDENCE",
        }
```

---

## 8. RLHF Export & Fine-Tuning Loop

### 8.1 `services/socketio_emitter.py` — Push-Back to Node Server

```python
# app/services/socketio_emitter.py

import httpx
import structlog
from app.core.config import get_settings
from app.schemas.sign import BoundingBox

logger = structlog.get_logger()


class SocketIOEmitter:
    """
    HTTP client for pushing inference events to the Node.js server,
    which then relays them to connected Socket.io clients.

    The ML service does NOT connect directly to clients — it pushes
    to the Node server's internal relay API, which forwards via Socket.io.
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.node_server_url
        self.headers = {
            "Authorization": f"Bearer {self.settings.ml_service_api_key}",
            "Content-Type": "application/json",
        }

    async def emit_inference_started(
        self,
        session_token: str,
        inference_id: str,
        modality: str,
    ) -> None:
        payload = {
            "event": "inference_started",
            "session_token": session_token,
            "inference_id": inference_id,
            "modality": modality,
        }
        await self._post("/api/v1/relay/emit", payload)

    async def emit_text_streamed(
        self,
        session_token: str,
        inference_id: str,
        partial_text: str,
        confidence_score: float,
        occlusion_detected: bool,
        bounding_box: BoundingBox | None,
    ) -> None:
        payload = {
            "event": "text_streamed",
            "session_token": session_token,
            "inference_id": inference_id,
            "partial_text": partial_text,
            "confidence_score": confidence_score,
            "occlusion_detected": occlusion_detected,
            "bounding_box": bounding_box.model_dump() if bounding_box else None,
        }
        await self._post("/api/v1/relay/emit", payload)

    async def emit_inference_complete(
        self,
        session_token: str,
        payload: dict,
    ) -> None:
        relay_payload = {"event": "inference_complete", "session_token": session_token, **payload}
        await self._post("/api/v1/relay/emit", relay_payload)

    async def push_rlhf_correction(
        self,
        inference_id: str,
        raw_prediction_text: str,
        final_corrected_text: str | None,
        was_edited: bool,
        was_rejected: bool,
        confidence_score: float,
        model_version: str,
        input_modality: str,
        inference_latency_ms: int,
        occlusion_detected: bool,
        frame_rate_avg: float | None = None,
    ) -> dict:
        """Push correction data to Node.js RLHF endpoint for DB persistence."""
        payload = {
            "inference_id": inference_id,
            "raw_prediction_text": raw_prediction_text,
            "final_corrected_text": final_corrected_text,
            "was_edited": was_edited,
            "was_rejected": was_rejected,
            "confidence_score": confidence_score,
            "model_version": model_version,
            "input_modality": input_modality,
            "inference_latency_ms": inference_latency_ms,
            "occlusion_detected": occlusion_detected,
            "frame_rate_avg": frame_rate_avg,
        }
        return await self._post("/api/v1/rlhf/corrections", payload)

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Node server HTTP error", status=e.response.status_code, path=path)
            return {}
        except httpx.RequestError as e:
            logger.error("Node server connection error", error=str(e), path=path)
            return {}
```

### 8.2 `services/rlhf_scheduler.py` — Automated Fine-Tune Trigger

```python
# app/services/rlhf_scheduler.py

import asyncio
import httpx
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()


class RLHFScheduler:
    """
    Background scheduler that:
      1. Periodically queries the Node server for GOLD/SILVER tier RLHF export data
      2. Evaluates whether fine-tuning thresholds are met
      3. Triggers fine-tuning script if thresholds exceeded
      4. Marks exported records as processed

    Runs on a configurable interval (default: every 24 hours).
    """

    def __init__(self):
        self.settings = get_settings()
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_export_timestamp: str | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("RLHF scheduler started", interval_hours=self.settings.rlhf_scheduler_interval_hours)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RLHF scheduler stopped")

    async def _run_loop(self) -> None:
        interval_s = self.settings.rlhf_scheduler_interval_hours * 3600
        while self._running:
            try:
                await self._run_rlhf_cycle()
            except Exception as e:
                logger.error("RLHF cycle failed", error=str(e))
            await asyncio.sleep(interval_s)

    async def _run_rlhf_cycle(self) -> None:
        """One full RLHF export and evaluation cycle."""
        logger.info("Starting RLHF export cycle")

        records = await self._fetch_rlhf_export()
        if not records:
            logger.info("No new RLHF records to process")
            return

        gold_silver = [r for r in records if r.get("quality_tier") in ("GOLD", "SILVER")]
        bronze = [r for r in records if r.get("quality_tier") == "BRONZE"]

        logger.info(
            "RLHF export fetched",
            total=len(records),
            gold_silver=len(gold_silver),
            bronze=len(bronze),
        )

        # Evaluate fine-tuning trigger conditions
        should_fine_tune = self._should_trigger_fine_tune(records)

        if should_fine_tune:
            logger.info("Fine-tune threshold met — preparing training data")
            await self._prepare_and_trigger_fine_tune(records)
        else:
            logger.info("Fine-tune threshold not yet met — saving for accumulation")
            self._save_export_locally(records)

        # Mark records as exported in Node DB
        await self._mark_exported(records)

    async def _fetch_rlhf_export(self) -> list[dict]:
        """Fetch unexported RLHF records from Node.js server."""
        settings = self.settings
        params = {
            "tier": "GOLD,SILVER,BRONZE",
            "limit": 2000,
            "modality": "SIGN",   # Focus on sign data for Siformer
        }
        if self._last_export_timestamp:
            params["since"] = self._last_export_timestamp

        headers = {
            "Authorization": f"Bearer {settings.ml_service_api_key}",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.node_server_url}/api/v1/rlhf/export",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        self._last_export_timestamp = data.get("exported_at")
        return data.get("records", [])

    def _should_trigger_fine_tune(self, records: list[dict]) -> bool:
        """
        Evaluate whether fine-tuning should be triggered.

        Conditions (either triggers):
          1. Total GOLD+SILVER+BRONZE records >= threshold (default: 500)
          2. Average edit_delta_chars for BRONZE tier exceeds threshold (default: 15)
        """
        settings = self.settings

        # Condition 1: Volume threshold
        if len(records) >= settings.rlhf_export_threshold:
            logger.info("Volume threshold met", count=len(records))
            return True

        # Condition 2: Average edit delta threshold
        bronze_records = [r for r in records if r.get("quality_tier") == "BRONZE"]
        if bronze_records:
            avg_delta = np.mean([
                r.get("edit_delta_chars", 0) for r in bronze_records
            ])
            if avg_delta >= settings.rlhf_edit_delta_threshold:
                logger.info("Edit delta threshold met", avg_delta=avg_delta)
                return True

        return False

    async def _prepare_and_trigger_fine_tune(self, records: list[dict]) -> None:
        """
        Save training data to disk and trigger fine-tuning pipeline.

        In production, this would submit a job to a GPU cluster (e.g., Slurm, K8s Job).
        In development, runs the training script directly.
        """
        output_dir = Path("data/rlhf_training") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSONL training data
        with open(output_dir / "corrections.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Training data saved", path=str(output_dir), count=len(records))

        # Trigger fine-tuning (async subprocess in production)
        # In a production K8s environment, this would create a Job manifest
        proc = await asyncio.create_subprocess_exec(
            "python",
            "scripts/train_siformer.py",
            "--rlhf_data", str(output_dir / "corrections.jsonl"),
            "--output_dir", "app/models/siformer/weights",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            logger.info("Fine-tuning completed successfully")
        else:
            logger.error("Fine-tuning failed", stderr=stderr.decode())

    def _save_export_locally(self, records: list[dict]) -> None:
        """Accumulate records locally for next trigger evaluation."""
        cache_dir = Path("data/rlhf_cache")
        cache_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(cache_dir / f"export_{timestamp}.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _mark_exported(self, records: list[dict]) -> None:
        """Notify Node server to mark these records as exported_for_training=true."""
        if not records:
            return
        settings = self.settings
        inference_ids = [r.get("inference_id") for r in records if r.get("inference_id")]

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{settings.node_server_url}/api/v1/rlhf/mark-exported",
                json={"inference_ids": inference_ids},
                headers={"Authorization": f"Bearer {settings.ml_service_api_key}"},
            )
```

---

## 9. Socket.io Event Contract Implementation

### 9.1 `schemas/sign.py`

```python
# app/schemas/sign.py

from pydantic import BaseModel, Field
from typing import Optional


class BoundingBox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Left edge, normalized 0–1")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge, normalized 0–1")
    width: float = Field(..., ge=0.0, le=1.0, description="Width, normalized 0–1")
    height: float = Field(..., ge=0.0, le=1.0, description="Height, normalized 0–1")
    is_occluded: bool = False
    occlusion_score: float = Field(0.0, ge=0.0, le=1.0)


class SignInferenceRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64-encoded JPEG frame")
    session_token: str
    inference_id: Optional[str] = None


class SignInferenceResponse(BaseModel):
    inference_id: str
    session_token: str
    raw_prediction_text: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    class_idx: int
    model_version: str
    inference_latency_ms: int
    occlusion_detected: bool


class FrameInferenceResult(BaseModel):
    """Per-frame result (not full Siformer — just MediaPipe + occlusion)."""
    landmarks: object              # FrameLandmarks dataclass
    bounding_box: Optional[BoundingBox]
    frame_confidence: float
    occlusion_detected: bool

    class Config:
        arbitrary_types_allowed = True
```

### 9.2 Complete Socket Event Flow (Summary)

```
CLIENT                  ML SERVICE              NODE SERVER             CLIENT (Receiver)
  │                         │                       │                       │
  │──── WS /v1/sign/stream ─►│                       │                       │
  │     { session_token }    │                       │                       │
  │                         │──POST /relay/emit ────►│                       │
  │                         │  inference_started     │──WS: inference_started►│
  │◄── { inference_started }─│                       │                       │
  │                         │                       │                       │
  │──── binary frame (JPEG) ►│                       │                       │
  │                         │ [MediaPipe extract]    │                       │
  │◄── { text_streamed,      │                       │                       │
  │      bounding_box,       │──POST /relay/emit ────►│                       │
  │      confidence }        │  text_streamed         │                       │
  │                         │                       │                       │
  │  [... more frames ...]  │                       │                       │
  │                         │                       │                       │
  │──── { type: end_signing }►│                       │                       │
  │                         │ [Siformer infer]       │                       │
  │◄── { inference_complete }─│                       │                       │
  │                         │──POST /relay/emit ────►│                       │
  │                         │  inference_complete     │──WS: text_streamed ───►│
  │                         │                       │──WS: inference_complete►│
  │                         │                       │                       │
  │  [User edits + sends]   │                       │                       │
  │──── POST /rlhf/correct ─►│  (via Node relay)     │                       │
  │                         │──POST /rlhf/correct ──►│                       │
  │                         │                        │ [DB: ai_inference_log]│
```

---

## 10. Testing & Benchmarking

### 10.1 `tests/test_sign_inference.py`

```python
# tests/test_sign_inference.py

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.siformer.architecture import Siformer
from app.services.sign_inference import SignInferenceService
from app.services.landmark_processor import NUM_JOINTS, FEATURE_DIM


@pytest.fixture
def dummy_model():
    """Lightweight Siformer for fast testing (tiny dimensions)."""
    return Siformer(
        num_joints=NUM_JOINTS,
        d_model=64,
        n_spatial=2,
        n_temporal=2,
        n_heads=4,
        ffn_dim=128,
        num_classes=10,
        seq_len=16,
    )


@pytest.fixture
def dummy_sequence():
    """Random normalized skeleton sequence (16 frames, 75 joints, 3 coords)."""
    return torch.randn(1, 16, NUM_JOINTS, FEATURE_DIM)


class TestSiformerArchitecture:
    def test_forward_pass_shape(self, dummy_model, dummy_sequence):
        dummy_model.eval()
        with torch.no_grad():
            logits, attn = dummy_model(dummy_sequence)
        assert logits.shape == (1, 10), f"Expected (1, 10), got {logits.shape}"

    def test_predict_returns_valid_confidence(self, dummy_model, dummy_sequence):
        class_idx, confidence, probs = dummy_model.predict(dummy_sequence)
        assert 0 <= class_idx < 10
        assert 0.0 <= confidence <= 1.0
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities must sum to 1"

    def test_padding_mask_applied(self, dummy_model):
        """Verify padded frames don't dominate the output."""
        x = torch.randn(1, 16, NUM_JOINTS, FEATURE_DIM)
        mask = torch.tensor([[False] * 8 + [True] * 8])  # Last 8 frames padded
        logits_masked, _ = dummy_model(x, padding_mask=mask)
        logits_unmasked, _ = dummy_model(x)
        assert not torch.allclose(logits_masked, logits_unmasked), \
            "Padding mask should affect output"

    def test_batch_inference(self, dummy_model):
        batch = torch.randn(8, 16, NUM_JOINTS, FEATURE_DIM)
        logits, _ = dummy_model(batch)
        assert logits.shape == (8, 10)


class TestLandmarkProcessor:
    def test_zero_output_on_blank_frame(self):
        from app.services.landmark_processor import LandmarkProcessor
        processor = LandmarkProcessor(model_complexity=0)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_frame(blank_frame)

        assert result.raw_vector.shape == (NUM_JOINTS, FEATURE_DIM)
        assert not result.left_hand_visible
        assert not result.right_hand_visible
        assert result.occlusion_score == 1.0
        processor.close()

    def test_normalization_clips_outliers(self):
        from app.services.landmark_processor import LandmarkProcessor
        processor = LandmarkProcessor(model_complexity=0)
        # Inject extreme coordinates
        coords = np.full((NUM_JOINTS, FEATURE_DIM), 100.0, dtype=np.float32)
        anchor = np.zeros(FEATURE_DIM)
        normalized = processor._normalize(coords, anchor, 1.0)
        assert np.all(normalized <= 2.0)
        assert np.all(normalized >= -2.0)
        processor.close()


class TestConfidenceAggregation:
    def test_high_confidence_peaked_distribution(self):
        service = MagicMock(spec=SignInferenceService)
        service._aggregate_confidence = SignInferenceService._aggregate_confidence.__get__(service)

        # Peaked distribution → high confidence
        probs = torch.zeros(300)
        probs[42] = 0.95
        probs[10] = 0.03
        probs[100] = 0.02
        confidence = service._aggregate_confidence(probs, top_k=3)
        assert confidence > 0.85, f"Peaked dist should give high confidence, got {confidence}"

    def test_low_confidence_uniform_distribution(self):
        service = MagicMock(spec=SignInferenceService)
        service._aggregate_confidence = SignInferenceService._aggregate_confidence.__get__(service)

        # Uniform distribution → low confidence
        probs = torch.ones(300) / 300
        confidence = service._aggregate_confidence(probs, top_k=3)
        assert confidence < 0.10, f"Uniform dist should give low confidence, got {confidence}"
```

### 10.2 Latency Benchmarking

```python
# scripts/benchmark_latency.py
"""
Benchmarks end-to-end inference latency for Siformer and Whisper.
Run: python scripts/benchmark_latency.py --n_runs 100
"""

import time
import argparse
import numpy as np
import torch
from pathlib import Path

from app.models.siformer.loader import SiformerLoader
from app.models.whisper.loader import WhisperLoader
from app.core.config import get_settings


def benchmark_siformer(n_runs: int = 100) -> dict:
    settings = get_settings()
    model = SiformerLoader.get_model()
    device = torch.device(settings.model_device)

    x = torch.randn(1, settings.siformer_sequence_len, 75, 3, device=device)
    latencies = []

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            model(x)

    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "model": "Siformer",
        "n_runs": n_runs,
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
    }


def benchmark_whisper(n_runs: int = 20) -> dict:
    import whisper as w
    settings = get_settings()
    model = WhisperLoader.get_model()

    # 3-second dummy audio at 16kHz
    audio = np.zeros(16000 * 3, dtype=np.float32)
    latencies = []

    for _ in range(n_runs):
        start = time.perf_counter()
        model.transcribe(audio, language="id", fp16=torch.cuda.is_available())
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "model": f"Whisper-{settings.whisper_model_size}",
        "n_runs": n_runs,
        "mean_ms": round(np.mean(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()

    print("\n── Siformer Latency ──")
    results = benchmark_siformer(args.n_runs)
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n── Whisper Latency ──")
    results = benchmark_whisper(min(args.n_runs, 20))
    for k, v in results.items():
        print(f"  {k}: {v}")
```

### 10.3 Target Performance Metrics

| Metric | Target | Notes |
|---|---|---|
| Siformer P50 latency | < 30ms | GPU inference, seq_len=64 |
| Siformer P99 latency | < 80ms | Under concurrent load |
| MediaPipe per-frame | < 20ms | Including landmark extraction |
| Whisper 3s audio (medium) | < 800ms | Indonesian, GPU |
| Whisper 3s audio (large) | < 1500ms | Indonesian, GPU |
| WebSocket frame round-trip | < 50ms | Frame → bounding box update |
| RLHF push-back latency | < 200ms | Non-blocking async |

---

## 11. Deployment & GPU Configuration

### 11.1 `Dockerfile`

```dockerfile
# apps/ml-service/Dockerfile

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for MediaPipe + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Pre-download Whisper model at build time to avoid runtime downloads
ARG WHISPER_MODEL_SIZE=large
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL_SIZE}')"

# Model weights mounted at runtime (see docker-compose volumes)
VOLUME /app/models

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "h11"]
```

> **Note on workers:** Use `--workers 1` with GPU to avoid CUDA context conflicts. For CPU deployment, `--workers 4` is safe.

### 11.2 GPU Memory Requirements

| Configuration | VRAM Required | Recommended GPU |
|---|---|---|
| Siformer (d_model=256) + Whisper medium | ~6 GB | NVIDIA RTX 3060 / T4 |
| Siformer (d_model=512) + Whisper large-v3 | ~12 GB | NVIDIA RTX 3080 / A10G |
| Full production (large-v3 + d_model=512, fp16) | ~8 GB (with fp16) | NVIDIA T4 / A10G |

### 11.3 Half-Precision Inference (Production Optimization)

```python
# In SiformerLoader._load() — add after model.to(device):

if device.type == "cuda":
    model = model.half()   # Convert to fp16 for ~2x speedup, ~50% VRAM reduction
    # Note: input tensors must also be .half() during inference

# In SignInferenceService.infer_buffer():
if self.device.type == "cuda":
    x = x.half()   # Match model dtype
```

### 11.4 ONNX Export for Production

```python
# scripts/export_onnx.py

import torch
from pathlib import Path
from app.models.siformer.loader import SiformerLoader
from app.core.config import get_settings

def export_to_onnx(output_path: str = "models/siformer/siformer_bisindo.onnx"):
    settings = get_settings()
    model = SiformerLoader.get_model()
    model.eval()

    dummy_input = torch.zeros(
        1, settings.siformer_sequence_len, 75, 3,
        device=next(model.parameters()).device
    )

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["skeleton_sequence"],
        output_names=["logits", "attention_weights"],
        dynamic_axes={
            "skeleton_sequence": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to {output_path}")
    print("Validate with: python -c \"import onnx; onnx.checker.check_model('{}')\"".format(output_path))

if __name__ == "__main__":
    export_to_onnx()
```

---

## Appendix A — BISINDO Dataset Integration Notes

### Rhiosutoyo/BISINDO-Hand-Sign-Detection-Dataset
Used for MediaPipe calibration and alphabet-level static sign classification baseline. Load via `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("Rhiosutoyo/BISINDO-Hand-Sign-Detection-Dataset")
# Use for initial hand detector confidence tuning
```

### Roboflow BISINDO Project L92HB
Export via Roboflow API for bounding box-annotated images. Use to validate `OcclusionDetector` accuracy against ground-truth hand positions.

### ademaulana/CNN-BISINDO
Load as baseline comparison:
```python
from transformers import AutoModelForImageClassification
baseline = AutoModelForImageClassification.from_pretrained("ademaulana/CNN-BISINDO")
# Compare top-1 accuracy on static frames vs Siformer on temporal sequences
```

### Mendeley BISINDO Dataset
Primary training source. After downloading, run:
```bash
python scripts/preprocess_dataset.py \
    --input_dir data/raw/mendeley_bisindo \
    --output_dir data/processed \
    --seq_len 64 \
    --stride 32
```

---

*End of Document — SLOP ML Service Implementation Guide v1.0.0*
