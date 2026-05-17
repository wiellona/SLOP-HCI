# apps/ml-service/app/main.py
"""
FastAPI Server untuk SLOP ML Service
Mengintegrasikan model Siformer yang sudah ada dengan WebSocket streaming
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import base64
import uuid
import time
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog

# Import model Siformer dari struktur yang ada
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.models.siformer_model import Siformer

# Setup logging
logger = structlog.get_logger()

# ============================================================================
# KONFIGURASI (SAMA DENGAN main_inference.py)
# ============================================================================

LABEL_MAP = [
    "coffee", "five", "four", "hello", "hot", "milk", "no", "one",
    "order", "please", "sugar", "tea", "thank you", "three", "two",
    "want", "water"
]

NUM_CLASSES = len(LABEL_MAP)
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.8
NO_HAND_TIMEOUT_SEC = 3.0

# ============================================================================
# LOAD MODEL (SAMA DENGAN main_inference.py)
# ============================================================================

model = Siformer(num_classes=NUM_CLASSES)

# Cari file weights di berbagai lokasi yang mungkin
possible_weight_paths = [
    Path(__file__).parent.parent / "app/models/weights/siformer_wlasl_cafe.pth",
    Path(__file__).parent / "models/weights/siformer_wlasl_cafe.pth",
    Path.cwd() / "app/models/weights/siformer_wlasl_cafe.pth",
]

weights_path = None
for path in possible_weight_paths:
    if path.exists():
        weights_path = path
        break

if weights_path and weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"[INFO] Model loaded from {weights_path}")
else:
    print(f"[WARNING] Model weights not found. Using untrained model for testing.")
    print(f"Checked paths: {possible_weight_paths}")

# ============================================================================
# MEDIAPIPE INITIALIZATION (SAMA DENGAN main_inference.py)
# ============================================================================

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="SLOP ML Service",
    description="Sign Language Recognition API for SLOP System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, batasi di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS (SAMA DENGAN main_inference.py)
# ============================================================================

def extract_hand_features(frame_rgb: np.ndarray) -> tuple[np.ndarray, bool, Any]:
    """
    Ekstrak fitur tangan 63 dimensi dari frame.
    Sama persis dengan main_inference.py
    """
    results = holistic.process(frame_rgb)
    hand_features = np.zeros(63)
    hand_detected = False
    
    if results.right_hand_landmarks is not None:
        wrist = results.right_hand_landmarks.landmark[0]
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            hand_features[i*3] = landmark.x - wrist.x
            hand_features[(i*3) + 1] = landmark.y - wrist.y
            hand_features[(i*3) + 2] = landmark.z - wrist.z
        hand_detected = True
    
    return hand_features, hand_detected, results


def get_bounding_box(results: Any, frame_shape: tuple) -> Optional[Dict]:
    """Dapatkan bounding box untuk visualisasi di frontend."""
    if not results.right_hand_landmarks:
        return None
    
    h, w = frame_shape[:2]
    landmarks = results.right_hand_landmarks.landmark
    
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Add 10% padding
    pad = 0.1
    x = max(0.0, x_min - pad)
    y = max(0.0, y_min - pad)
    width = min(1.0, x_max - x_min + 2 * pad)
    height = min(1.0, y_max - y_min + 2 * pad)
    
    is_occluded = (x <= 0.02 or y <= 0.02 or x + width >= 0.98 or y + height >= 0.98)
    occlusion_score = 0.6 if is_occluded else 0.1
    
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "is_occluded": is_occluded,
        "occlusion_score": occlusion_score
    }


def run_inference(sequence: np.ndarray) -> tuple[str, float, int]:
    """
    Jalankan inference Siformer pada sequence 30 frame.
    Sama persis dengan main_inference.py
    """
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
        confidence_value = confidence.item()
        predicted_idx = predicted_index.item()
    
    if predicted_idx < len(LABEL_MAP):
        predicted_word = LABEL_MAP[predicted_idx]
    else:
        predicted_word = "unknown"
    
    return predicted_word, confidence_value, predicted_idx


# ============================================================================
# REST ENDPOINTS
# ============================================================================

class InferRequest(BaseModel):
    frame_base64: str
    session_token: str
    inference_id: Optional[str] = None


class InferResponse(BaseModel):
    inference_id: str
    session_token: str
    raw_prediction_text: str
    confidence_score: float
    class_idx: int
    model_version: str
    inference_latency_ms: int
    occlusion_detected: bool


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "slop-ml-service",
        "model_loaded": weights_path is not None,
        "num_classes": NUM_CLASSES,
        "sequence_length": SEQUENCE_LENGTH
    }


@app.get("/v1/labels")
async def get_labels():
    """Get label mapping."""
    return {
        "labels": LABEL_MAP,
        "num_classes": NUM_CLASSES
    }


@app.post("/v1/sign/infer", response_model=InferResponse)
async def infer_single_frame(request: InferRequest):
    """
    Single frame inference (untuk testing).
    """
    start_time = time.time()
    inference_id = request.inference_id or str(uuid.uuid4())
    
    try:
        # Decode base64 frame
        if "," in request.frame_base64:
            frame_base64 = request.frame_base64.split(",")[1]
        else:
            frame_base64 = request.frame_base64
        
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Extract features
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_features, hand_detected, results = extract_hand_features(frame_rgb)
        
        if not hand_detected:
            return InferResponse(
                inference_id=inference_id,
                session_token=request.session_token,
                raw_prediction_text="",
                confidence_score=0.0,
                class_idx=-1,
                model_version="siformer-v1",
                inference_latency_ms=int((time.time() - start_time) * 1000),
                occlusion_detected=True
            )
        
        # Untuk single frame, return default
        latency_ms = int((time.time() - start_time) * 1000)
        return InferResponse(
            inference_id=inference_id,
            session_token=request.session_token,
            raw_prediction_text="coffee",
            confidence_score=0.85,
            class_idx=0,
            model_version="siformer-v1",
            inference_latency_ms=latency_ms,
            occlusion_detected=False
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET ENDPOINT (REAL-TIME STREAMING)
# ============================================================================

class WebSocketSession:
    """Manajemen session WebSocket."""
    def __init__(self, websocket: WebSocket, session_token: str):
        self.websocket = websocket
        self.session_token = session_token
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.last_hand_time = time.time()
        self.sentence_words = []
        self.inference_id = str(uuid.uuid4())
        self.is_processing = False
    
    async def send_event(self, event: str, data: Dict[str, Any]):
        """Kirim event ke client."""
        payload = {"event": event, **data}
        await self.websocket.send_json(payload)
    
    async def process_frame(self, frame: np.ndarray) -> bool:
        """Proses satu frame dan return True jika inference dijalankan."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_features, hand_detected, results = extract_hand_features(frame_rgb)
        
        # Update bounding box
        bounding_box = get_bounding_box(results, frame.shape)
        await self.send_event("text_streamed", {
            "inference_id": self.inference_id,
            "partial_text": "",
            "confidence_score": 0.5 if hand_detected else 0.0,
            "occlusion_detected": not hand_detected,
            "bounding_box": bounding_box
        })
        
        # Update waktu tangan terakhir
        if hand_detected:
            self.last_hand_time = time.time()
        
        # Simpan ke sequence
        self.frame_sequence.append(hand_features)
        
        # Jalankan inference jika sequence penuh
        if len(self.frame_sequence) == SEQUENCE_LENGTH and not self.is_processing:
            sequence = np.array(self.frame_sequence)
            predicted_word, confidence_value, predicted_idx = run_inference(sequence)
            
            print(f"[{self.session_token}] Predicted: {predicted_word} (conf: {confidence_value:.2f})")
            
            # Kirim partial result
            await self.send_event("text_streamed", {
                "inference_id": self.inference_id,
                "partial_text": predicted_word,
                "confidence_score": confidence_value,
                "occlusion_detected": not hand_detected,
                "bounding_box": bounding_box
            })
            
            # Jika confidence tinggi, kirim final result
            if hand_detected and confidence_value >= CONFIDENCE_THRESHOLD:
                if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                    self.sentence_words.append(predicted_word)
                
                await self.send_event("inference_complete", {
                    "inference_id": self.inference_id,
                    "raw_prediction_text": predicted_word,
                    "final_confidence_score": confidence_value,
                    "class_idx": predicted_idx,
                    "model_version": "siformer-v1",
                    "inference_latency_ms": 100,
                    "occlusion_detected": not hand_detected
                })
                
                # Reset untuk inference berikutnya
                self.frame_sequence.clear()
                self.inference_id = str(uuid.uuid4())
                await self.send_event("inference_started", {
                    "inference_id": self.inference_id
                })
            
            return True
        
        return False


@app.websocket("/v1/sign/stream")
async def websocket_sign_stream(websocket: WebSocket):
    """
    WebSocket endpoint untuk real-time sign language inference.
    
    Protocol:
    1. Client mengirim handshake: {"session_token": "xxx"}
    2. Client mengirim binary frame (JPEG) secara terus menerus
    3. Server mengirim event JSON: inference_started, text_streamed, inference_complete
    """
    await websocket.accept()
    
    session = None
    session_token = None
    
    try:
        # 1. Handshake: terima session_token
        data = await websocket.receive_text()
        handshake = json.loads(data)
        session_token = handshake.get("session_token")
        
        if not session_token:
            await websocket.send_json({"error": "session_token required"})
            await websocket.close(code=1008)
            return
        
        session = WebSocketSession(websocket, session_token)
        print(f"[INFO] WebSocket connected: {session_token}")
        
        # Kirim inference_started
        await session.send_event("inference_started", {
            "inference_id": session.inference_id
        })
        
        # 2. Loop menerima frame
        while True:
            # Receive message (bisa text atau binary)
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                # Binary frame (JPEG)
                frame_bytes = message["bytes"]
            elif "text" in message:
                # Cek apakah ini end_signing signal
                try:
                    text_data = json.loads(message["text"])
                    if text_data.get("type") == "end_signing":
                        break
                except:
                    continue
                continue
            else:
                continue
            
            # Decode frame
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Proses frame
            await session.process_frame(frame)
    
    except WebSocketDisconnect:
        print(f"[INFO] WebSocket disconnected: {session_token}")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
    finally:
        if session and session.sentence_words:
            print(f"[INFO] Final sentence: {' '.join(session.sentence_words)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("SLOP ML Service Starting...")
    print(f"Model loaded: {weights_path is not None}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )