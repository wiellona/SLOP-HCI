from io import BytesIO
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import json
import numpy as np
import cv2
import time
from loguru import logger
from collections import deque
import torch
import mediapipe as mp
import asyncio
import os

try:
    from PIL import Image
except ImportError:
    Image = None

from app.core.config import get_settings
from app.hand_preprocess import (
    FEATURE_SIZE_ONE_HAND,
    FEATURE_SIZE_TWO_HANDS,
    PredictionFilter,
    compute_motion_score,
    normalize_hand_landmarks,
    normalize_two_hands,
)
from app.models.siformer_model import Siformer

router = APIRouter()
settings = get_settings()

DEFAULT_LABEL_MAP = [
    "allergy",
    "change",
    "coffee",
    "five",
    "four",
    "hello",
    "hot",
    "me",
    "milk",
    "no",
    "one",
    "order",
    "peanut butter",
    "please",
    "sugar",
    "tea",
    "thank you",
    "three",
    "two",
    "want",
    "water",
    "with",
]


def _load_label_map(path: str, fallback: List[str]) -> List[str]:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                labels = json.load(handle)
                if isinstance(labels, list) and labels:
                    return labels
        except Exception as exc:
            logger.warning(f"Failed to load labels from {path}: {exc}")
    return fallback


def _load_translation_map(path: str) -> Dict[str, str]:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning(f"Failed to load translation map from {path}: {exc}")
    return {}


def _resolve_device(device_hint: Optional[str]) -> torch.device:
    if device_hint:
        if device_hint.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_hint)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: str) -> Optional[Dict[str, torch.Tensor]]:
    if not path or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return None


def _infer_num_joints(state_dict: Dict[str, torch.Tensor], in_channels: int = 3) -> Optional[int]:
    weight = state_dict.get("input_projection.weight") if state_dict else None
    if weight is None or weight.ndim != 2:
        return None
    input_dim = weight.shape[1]
    if input_dim % in_channels != 0:
        return None
    return int(input_dim / in_channels)


def _resolve_model_config(
    state_dict: Optional[Dict[str, torch.Tensor]],
    use_two_hands: bool,
    in_channels: int = 3,
) -> tuple[int, bool, int]:
    num_joints = 42 if use_two_hands else 21
    inferred = _infer_num_joints(state_dict or {}, in_channels=in_channels)
    if inferred in (21, 42) and inferred != num_joints:
        logger.warning(
            "Model weights indicate {} joints. Overriding use_two_hands={}.",
            inferred,
            use_two_hands,
        )
        num_joints = inferred
        use_two_hands = num_joints == 42
    feature_size = num_joints * in_channels
    return num_joints, use_two_hands, feature_size


def _translate_tokens(tokens: List[str], translation_map: Dict[str, str]) -> List[str]:
    return [translation_map.get(token, token) for token in tokens if token]


def _landmarks_to_bbox(landmarks: Optional[object], margin: float = 0.02) -> Optional[Dict[str, float]]:
    if landmarks is None:
        return None

    coords = landmarks.landmark if hasattr(landmarks, "landmark") else None
    if not coords:
        return None

    xs = [lm.x for lm in coords]
    ys = [lm.y for lm in coords]

    if not xs or not ys:
        return None

    min_x = max(0.0, min(xs) - margin)
    min_y = max(0.0, min(ys) - margin)
    max_x = min(1.0, max(xs) + margin)
    max_y = min(1.0, max(ys) + margin)

    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)

    edge_distance = min(min_x, min_y, 1.0 - max_x, 1.0 - max_y)
    occlusion_score = 0.0
    if edge_distance < margin:
        occlusion_score = min(1.0, (margin - edge_distance) / margin)

    return {
        "x": min_x,
        "y": min_y,
        "width": width,
        "height": height,
        "is_occluded": occlusion_score > 0.0,
        "occlusion_score": occlusion_score,
    }


def _merge_bboxes(bboxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not bboxes:
        return None

    min_x = min(box["x"] for box in bboxes)
    min_y = min(box["y"] for box in bboxes)
    max_x = max(box["x"] + box["width"] for box in bboxes)
    max_y = max(box["y"] + box["height"] for box in bboxes)

    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)

    occlusion_score = max(box.get("occlusion_score", 0.0) for box in bboxes)
    is_occluded = any(box.get("is_occluded") for box in bboxes)

    return {
        "x": min_x,
        "y": min_y,
        "width": width,
        "height": height,
        "is_occluded": is_occluded,
        "occlusion_score": occlusion_score,
    }


LABEL_MAP = _load_label_map(settings.sign_labels_path, DEFAULT_LABEL_MAP)
TRANSLATION_MAP = _load_translation_map(settings.sign_translation_map_path)

NUM_CLASSES = len(LABEL_MAP)
SEQUENCE_LENGTH = settings.sign_sequence_length
CONFIDENCE_THRESHOLD = settings.sign_confidence_threshold
REQUIRE_BOTH_HANDS = settings.sign_require_both_hands

# Mirror main_inference.py filtering defaults for gesture stability.
FILTER_WINDOW_SIZE = 5
FILTER_MIN_CONFIDENCE = 0.65
FILTER_STABLE_FRAMES = 3
FILTER_COOLDOWN_FRAMES = 10
FILTER_MOTION_THRESHOLD = 0.08

logger.info(f"Number of classes: {NUM_CLASSES}")
logger.info(f"Label map: {LABEL_MAP}")

state_dict = _load_checkpoint(settings.sign_model_path)
NUM_JOINTS, USE_TWO_HANDS, FEATURE_SIZE = _resolve_model_config(
    state_dict,
    settings.sign_use_two_hands,
)
REQUIRE_BOTH_HANDS = REQUIRE_BOTH_HANDS if USE_TWO_HANDS else False

device = _resolve_device(settings.sign_device)
model = None

try:
    model = Siformer(
        num_joints=NUM_JOINTS,
        num_classes=NUM_CLASSES,
        num_frames=SEQUENCE_LENGTH,
    )
    if not state_dict:
        raise FileNotFoundError(f"Model weights not found at {settings.sign_model_path}")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    model = model.to(device)
    model.eval()
    logger.info(
        "Model ready. device={}, num_joints={}, feature_size={}",
        device,
        NUM_JOINTS,
        FEATURE_SIZE,
    )

    dummy_input = torch.randn(1, SEQUENCE_LENGTH, FEATURE_SIZE, device=device)
    with torch.no_grad():
        test_output = model(dummy_input)
    logger.info(f"Model test successful, output shape: {test_output.shape}")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=settings.sign_min_detection_confidence,
    min_tracking_confidence=settings.sign_min_tracking_confidence,
)

class SignLanguageProcessor:
    def __init__(self):
        self.frame_sequences: Dict[str, deque] = {}
        self.sentence_buffers: Dict[str, list] = {}
        self.last_word_times: Dict[str, float] = {}
        self.last_hand_times: Dict[str, float] = {}
        self.last_predictions: Dict[str, str] = {}
        self.sentence_gap_ms = 2000
        self.no_hand_timeout_sec = 1.0
        self.frame_counters: Dict[str, int] = {}
        self.last_bounding_boxes: Dict[str, Optional[dict]] = {}
        self.prediction_filters: Dict[str, PredictionFilter] = {}
        self.prev_features: Dict[str, Optional[np.ndarray]] = {}
        
    def get_or_create_sequence(self, session_id: str) -> deque:
        if session_id not in self.frame_sequences:
            self.frame_sequences[session_id] = deque(maxlen=SEQUENCE_LENGTH)
            self.sentence_buffers[session_id] = []
            self.last_word_times[session_id] = time.time()
            self.last_hand_times[session_id] = time.time()
            self.last_predictions[session_id] = ""
            self.frame_counters[session_id] = 0
            self.last_bounding_boxes[session_id] = None
            self.prediction_filters[session_id] = PredictionFilter(
                window_size=FILTER_WINDOW_SIZE,
                min_confidence=FILTER_MIN_CONFIDENCE,
                stable_frames=FILTER_STABLE_FRAMES,
                cooldown_frames=FILTER_COOLDOWN_FRAMES,
                motion_threshold=FILTER_MOTION_THRESHOLD,
            )
            self.prev_features[session_id] = None
        return self.frame_sequences[session_id]

    def finalize_sentence(self, session_id: str) -> Optional[dict]:
        sentence_tokens = self.sentence_buffers.get(session_id, [])
        if not sentence_tokens:
            return None

        final_sentence = " ".join(sentence_tokens)
        translated_sentence = " ".join(_translate_tokens(sentence_tokens, TRANSLATION_MAP))

        self.sentence_buffers[session_id] = []
        self.last_predictions[session_id] = ""

        sequence = self.frame_sequences.get(session_id)
        if sequence is not None:
            sequence.clear()

        if session_id in self.prediction_filters:
            self.prediction_filters[session_id].reset()
        self.prev_features[session_id] = None

        return {
            "sentence_finalized": True,
            "predicted_word": "",
            "translated_word": "",
            "confidence_score": 0.0,
            "current_sentence": final_sentence,
            "translated_sentence": translated_sentence,
            "alternatives": [],
            "translated_alternatives": [],
            "inference_latency_ms": 0,
        }
    
    def decode_frame(self, frame_bytes: bytes):
        """Decode frame from bytes"""
        try:
            frame_start = time.perf_counter()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # If failed, try as PIL image when Pillow is available.
            if frame is None and Image is not None:
                image = Image.open(BytesIO(frame_bytes))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None
    
    def calculate_bounding_box(self, hand_landmarks, frame_shape):
        """Calculate normalized bounding box from hand landmarks"""
        h, w = frame_shape[:2]
        
        if not hand_landmarks:
            return None
        
        # Extract coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        if not x_coords or not y_coords:
            return None
        
        # Calculate bounding box with padding
        padding = 30
        min_x = max(0, int(min(x_coords) - padding))
        min_y = max(0, int(min(y_coords) - padding))
        max_x = min(w, int(max(x_coords) + padding))
        max_y = min(h, int(max(y_coords) + padding))
        
        # Normalize coordinates (0-1 range)
        normalized_box = {
            "x": min_x / w,
            "y": min_y / h,
            "width": (max_x - min_x) / w,
            "height": (max_y - min_y) / h,
            "is_occluded": False,
            "occlusion_score": 0.0
        }
        
        # Check for occlusion (if hand is near edge of frame)
        edge_threshold = 0.1
        if (normalized_box["x"] < edge_threshold or 
            normalized_box["y"] < edge_threshold or
            normalized_box["x"] + normalized_box["width"] > 1 - edge_threshold or
            normalized_box["y"] + normalized_box["height"] > 1 - edge_threshold):
            normalized_box["is_occluded"] = True
            normalized_box["occlusion_score"] = 0.7
        
        return normalized_box
    
    def extract_hand_features(self, hand_landmarks):
        """Extract hand landmarks features"""
        features = np.zeros(63)  # 21 landmarks * 3 (x,y,z)
        
        if hand_landmarks:
            # Use wrist as reference point
            wrist = hand_landmarks.landmark[0]
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i < 21:  # Ensure we don't exceed array bounds
                    features[i*3] = landmark.x - wrist.x
                    features[(i*3) + 1] = landmark.y - wrist.y
                    features[(i*3) + 2] = landmark.z - wrist.z
        
        return features
    
    def process_frame(self, session_id: str, frame_bytes: bytes) -> dict:
        try:
            frame_start = time.perf_counter()

            # Decode frame
            frame = self.decode_frame(frame_bytes)
            if frame is None:
                return {"error": "Failed to decode frame"}
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            right_hand = results.right_hand_landmarks
            left_hand = results.left_hand_landmarks
            hand_count = int(right_hand is not None) + int(left_hand is not None)

            if USE_TWO_HANDS:
                normalized = normalize_two_hands(
                    right_hand,
                    left_hand,
                    require_both_hands=REQUIRE_BOTH_HANDS,
                )
            else:
                primary_hand = right_hand or left_hand
                normalized = normalize_hand_landmarks(primary_hand)

            hand_features = (
                normalized
                if normalized is not None
                else np.zeros(FEATURE_SIZE, dtype=np.float32)
            )
            hand_detected = hand_count > 0

            sequence = self.get_or_create_sequence(session_id)

            motion_score = None
            if hand_detected:
                motion_score = compute_motion_score(
                    self.prev_features.get(session_id),
                    hand_features,
                )
                self.prev_features[session_id] = hand_features
            else:
                self.prev_features[session_id] = None
                if session_id in self.prediction_filters:
                    self.prediction_filters[session_id].reset()

            bboxes = []
            right_bbox = _landmarks_to_bbox(right_hand)
            if right_bbox:
                bboxes.append(right_bbox)
            left_bbox = _landmarks_to_bbox(left_hand)
            if left_bbox:
                bboxes.append(left_bbox)
            bounding_box = _merge_bboxes(bboxes)

            if hand_detected:
                self.last_hand_times[session_id] = time.time()

            if hand_detected:
                sequence.append(hand_features)
            self.frame_counters[session_id] += 1
            
            frame_processing_ms = int((time.perf_counter() - frame_start) * 1000)
            result = {
                "hand_detected": hand_detected,
                "hand_count": hand_count,
                "sequence_length": len(sequence),
                "bounding_box": bounding_box,
                "frame_processing_ms": frame_processing_ms,
            }
            
            # Run inference only while a hand is visible.
            if hand_detected and len(sequence) == SEQUENCE_LENGTH:
                inference_result = self.run_inference(sequence, session_id, motion_score)
                result.update(inference_result)

            if not hand_detected:
                last_hand_time = self.last_hand_times.get(session_id, time.time())
                if (time.time() - last_hand_time) >= self.no_hand_timeout_sec:
                    finalized_sentence = self.finalize_sentence(session_id)
                    if finalized_sentence:
                        result.update(finalized_sentence)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return {"error": str(e)}
    
    def run_inference(
        self,
        sequence: deque,
        session_id: str,
        motion_score: Optional[float] = None,
    ) -> dict:
        try:
            inference_start = time.perf_counter()
            sequence_matrix = np.asarray(sequence, dtype=np.float32)
            input_tensor = torch.from_numpy(sequence_matrix).unsqueeze(0).to(device)

            confidence_value = 0.0
            predicted_index = None
            probabilities = None
            probs_np: Optional[np.ndarray] = None

            if model is not None:
                with torch.inference_mode():
                    output_logits = model(input_tensor)
                    probabilities = torch.softmax(output_logits, dim=1)
                    confidence, predicted_index = torch.max(probabilities, dim=1)
                    confidence_value = confidence.item()
                    predicted_index = predicted_index.item()
                    probs_np = probabilities.detach().cpu().numpy().flatten()

            prediction_filter = self.prediction_filters.get(session_id)
            label_idx = None
            filtered_conf = None
            emit = False
            if prediction_filter is not None:
                label_idx, filtered_conf, emit = prediction_filter.update(
                    probs_np,
                    motion_score,
                )

            if label_idx is not None and label_idx < len(LABEL_MAP):
                predicted_word = LABEL_MAP[label_idx]
            else:
                predicted_word = ""

            if filtered_conf is not None:
                confidence_value = filtered_conf
            elif probs_np is None:
                confidence_value = 0.0

            if emit and predicted_word:
                if (not self.sentence_buffers[session_id] or
                        self.sentence_buffers[session_id][-1] != predicted_word):
                    self.sentence_buffers[session_id].append(predicted_word)
            
            sentence = " ".join(self.sentence_buffers[session_id])

            translated_word = TRANSLATION_MAP.get(predicted_word, predicted_word)
            translated_sentence = " ".join(
                _translate_tokens(self.sentence_buffers[session_id], TRANSLATION_MAP)
            )
            
            # Get alternative predictions
            alternatives = []
            if probabilities is not None and confidence_value < 0.85:
                top_probs, top_indices = torch.topk(probabilities, k=min(3, NUM_CLASSES))
                alternatives = [
                    LABEL_MAP[idx]
                    for idx in top_indices[0].tolist()
                    if LABEL_MAP[idx] != predicted_word and idx < len(LABEL_MAP)
                ][:3]

            translated_alternatives = [
                TRANSLATION_MAP.get(item, item) for item in alternatives
            ]

            inference_latency_ms = int((time.perf_counter() - inference_start) * 1000)
            
            return {
                "predicted_word": predicted_word,
                "translated_word": translated_word,
                "confidence_score": confidence_value,
                "current_sentence": sentence,
                "translated_sentence": translated_sentence,
                "alternatives": alternatives,
                "translated_alternatives": translated_alternatives,
                "inference_latency_ms": inference_latency_ms,
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return {
                "predicted_word": "",
                "translated_word": "",
                "confidence_score": 0,
                "current_sentence": "",
                "translated_sentence": "",
                "alternatives": [],
                "translated_alternatives": [],
                "inference_latency_ms": 0,
            }
    
    def clear_session(self, session_id: str):
        """Clear session data"""
        self.frame_sequences.pop(session_id, None)
        self.sentence_buffers.pop(session_id, None)
        self.last_word_times.pop(session_id, None)
        self.last_predictions.pop(session_id, None)
        self.frame_counters.pop(session_id, None)
        self.last_bounding_boxes.pop(session_id, None)

# Global processor
processor = SignLanguageProcessor()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        self.active_connections[session_id] = websocket
        logger.info(f"Client {session_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            processor.clear_session(session_id)
            logger.info(f"Client {session_id} disconnected. Remaining: {len(self.active_connections)}")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast(self, data: dict):
        """Broadcast to all connections"""
        for session_id in list(self.active_connections.keys()):
            await self.send_json(session_id, data)

manager = ConnectionManager()

@router.websocket("/stream")
async def sign_language_stream(websocket: WebSocket):
    session_id = None
    
    try:
        # Accept connection
        await websocket.accept()
        logger.info("WebSocket connection accepted, waiting for session token...")
        
        # Send ready message
        await websocket.send_json({
            "event": "ready",
            "message": "WebSocket connected, please send session token"
        })
        
        # Wait for session token with timeout
        try:
            init_data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            init_json = json.loads(init_data)
            session_id = init_json.get("session_token")
            
            if not session_id:
                session_id = f"session_{int(time.time())}_{id(websocket)}"
                logger.warning(f"No session token provided, using generated: {session_id}")
            else:
                logger.info(f"Session token received: {session_id}")
                
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session token")
            await websocket.close(code=1008, reason="Session token required")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await websocket.close(code=1008, reason="Invalid session token format")
            return
        
        # Register connection
        await manager.connect(websocket, session_id)
        
        # Send confirmation with model status
        await manager.send_json(session_id, {
            "event": "connected",
            "session_id": session_id,
            "message": "Connected to sign language service",
            "model_loaded": model is not None,
            "use_two_hands": USE_TWO_HANDS,
            "feature_size": FEATURE_SIZE,
            "device": str(device),
            "translation_enabled": bool(TRANSLATION_MAP),
        })
        
        logger.info(f"Session {session_id} ready, starting frame processing")
        
        # Frame processing variables
        frame_count = 0
        last_inference_time = time.monotonic()
        inference_interval = settings.sign_inference_interval_ms / 1000.0
        
        while True:
            try:
                # Receive binary frame data
                frame_data = await websocket.receive_bytes()
                frame_count += 1
                
                # Process frame at limited FPS
                current_time = time.monotonic()
                if current_time - last_inference_time >= inference_interval:
                    result = processor.process_frame(session_id, frame_data)
                    last_inference_time = current_time
                    
                    if result.get("error"):
                        await manager.send_json(session_id, {
                            "event": "error",
                            "error": result["error"]
                        })
                    else:
                        # Send tracking updates without streaming partial translation.
                        if not result.get("sentence_finalized"):
                            await manager.send_json(session_id, {
                                "event": "text_streamed",
                                "partial_text": result.get("predicted_word", ""),
                                "translated_partial_text": result.get("translated_word", ""),
                                "current_sentence": result.get("current_sentence", ""),
                                "translated_sentence": result.get("translated_sentence", ""),
                                "confidence_score": result.get("confidence_score", 0),
                                "inference_latency_ms": result.get("inference_latency_ms", 0),
                                "frame_processing_ms": result.get("frame_processing_ms", 0),
                                "hand_detected": result.get("hand_detected", False),
                                "hand_count": result.get("hand_count", 0),
                                "sequence_length": result.get("sequence_length", 0),
                                "bounding_box": result.get("bounding_box"),
                                "occlusion_detected": bool(
                                    (result.get("bounding_box") or {}).get("is_occluded")
                                ),
                            })
                        
                        # Send the full sentence only after the hand has been absent long enough.
                        if result.get("sentence_finalized"):
                            await manager.send_json(session_id, {
                                "event": "inference_complete",
                                "raw_prediction_text": result.get("predicted_word", ""),
                                "translated_word": result.get("translated_word", ""),
                                "final_confidence_score": result.get("confidence_score", 0),
                                "full_sentence": result.get("current_sentence", ""),
                                "translated_sentence": result.get("translated_sentence", ""),
                                "alternatives": result.get("alternatives", []),
                                "translated_alternatives": result.get("translated_alternatives", []),
                                "inference_latency_ms": result.get("inference_latency_ms", 0),
                            })
                
            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await manager.send_json(session_id, {
                    "event": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected before session initialization")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if session_id:
            manager.disconnect(session_id)

@router.get("/health")
async def sign_health():
    """Check if sign language service is working"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES,
        "labels": LABEL_MAP,
        "use_two_hands": USE_TWO_HANDS,
        "feature_size": FEATURE_SIZE,
        "device": str(device),
        "translation_enabled": bool(TRANSLATION_MAP),
        "sequence_length": SEQUENCE_LENGTH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "inference_interval_ms": settings.sign_inference_interval_ms,
    }