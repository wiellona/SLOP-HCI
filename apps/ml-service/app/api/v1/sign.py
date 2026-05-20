from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict
import json
import numpy as np
import cv2
import time
from loguru import logger
from collections import deque
import torch
import mediapipe as mp
import asyncio

router = APIRouter()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load Siformer model
from app.models.siformer_model import Siformer

LABEL_MAP = [
    "coffee", "five", "four", "hello", "hot", "milk", "no", "one",
    "order", "please", "sugar", "tea", "thank you", "three", "two",
    "want", "water", "ice"
]

NUM_CLASSES = len(LABEL_MAP)
logger.info(f"Number of classes: {NUM_CLASSES}")
logger.info(f"Label map: {LABEL_MAP}")

# Initialize model
model = Siformer(num_classes=NUM_CLASSES)

# Load weights
weights_path = "app/models/weights/siformer_wlasl_cafe.pth"
try:
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()
    logger.info(f"Model loaded successfully from {weights_path}")
    
    dummy_input = torch.randn(1, 30, 63)
    with torch.no_grad():
        test_output = model(dummy_input)
    logger.info(f"Model test successful, output shape: {test_output.shape}")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7

class SignLanguageProcessor:
    def __init__(self):
        self.frame_sequences: Dict[str, deque] = {}
        self.sentence_buffers: Dict[str, list] = {}
        self.last_word_times: Dict[str, float] = {}
        self.last_predictions: Dict[str, str] = {}
        self.sentence_gap_ms = 2000
        
    def get_or_create_sequence(self, session_id: str) -> deque:
        if session_id not in self.frame_sequences:
            self.frame_sequences[session_id] = deque(maxlen=SEQUENCE_LENGTH)
            self.sentence_buffers[session_id] = []
            self.last_word_times[session_id] = time.time()
            self.last_predictions[session_id] = ""
        return self.frame_sequences[session_id]
    
    def process_frame(self, session_id: str, frame_bytes: bytes) -> dict:
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "Failed to decode frame"}
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            hand_features = np.zeros(63)
            hand_detected = results.right_hand_landmarks is not None
            bounding_box = None
            
            if hand_detected:
                # Extract hand landmarks
                wrist = results.right_hand_landmarks.landmark[0]
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    hand_features[i*3] = landmark.x - wrist.x
                    hand_features[(i*3) + 1] = landmark.y - wrist.y
                    hand_features[(i*3) + 2] = landmark.z - wrist.z
                
                # Calculate bounding box from hand landmarks
                h, w = frame.shape[:2]
                x_coords = [lm.x * w for lm in results.right_hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in results.right_hand_landmarks.landmark]
                
                if x_coords and y_coords:
                    min_x = max(0, int(min(x_coords) - 20))
                    min_y = max(0, int(min(y_coords) - 20))
                    max_x = min(w, int(max(x_coords) + 20))
                    max_y = min(h, int(max(y_coords) + 20))
                    
                    bounding_box = {
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "is_occluded": False,
                        "occlusion_score": 0.0
                    }
            
            sequence = self.get_or_create_sequence(session_id)
            sequence.append(hand_features)
            
            result = {
                "hand_detected": hand_detected,
                "sequence_length": len(sequence),
                "bounding_box": bounding_box
            }
            
            if len(sequence) == SEQUENCE_LENGTH:
                inference_result = self.run_inference(sequence, session_id)
                result.update(inference_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"error": str(e)}
    
    def run_inference(self, sequence: deque, session_id: str) -> dict:
        try:
            sequence_matrix = np.array(sequence)
            input_tensor = torch.tensor(sequence_matrix, dtype=torch.float32).unsqueeze(0)
            
            if model is not None:
                with torch.no_grad():
                    output_logits = model(input_tensor)
                    probabilities = torch.softmax(output_logits, dim=1)
                    confidence, predicted_index = torch.max(probabilities, dim=1)
                    confidence_value = confidence.item()
                    predicted_index = predicted_index.item()
            else:
                confidence_value = 0.5
                predicted_index = np.random.randint(0, NUM_CLASSES)
            
            if predicted_index < len(LABEL_MAP):
                predicted_word = LABEL_MAP[predicted_index]
            else:
                predicted_word = ""
            
            current_time = time.time()
            last_time = self.last_word_times.get(session_id, current_time)
            last_pred = self.last_predictions.get(session_id, "")
            
            if confidence_value >= CONFIDENCE_THRESHOLD and predicted_word:
                if predicted_word != last_pred or (current_time - last_time) > self.sentence_gap_ms / 1000:
                    if current_time - last_time > self.sentence_gap_ms / 1000:
                        self.sentence_buffers[session_id] = [predicted_word]
                    else:
                        if (not self.sentence_buffers[session_id] or 
                            self.sentence_buffers[session_id][-1] != predicted_word):
                            self.sentence_buffers[session_id].append(predicted_word)
                    
                    self.last_word_times[session_id] = current_time
                    self.last_predictions[session_id] = predicted_word
            
            sentence = " ".join(self.sentence_buffers[session_id])
            
            alternatives = []
            if model is not None and confidence_value < 0.85:
                with torch.no_grad():
                    top_probs, top_indices = torch.topk(probabilities, k=min(3, NUM_CLASSES))
                    alternatives = [
                        LABEL_MAP[idx] for idx in top_indices[0].tolist() 
                        if LABEL_MAP[idx] != predicted_word and idx < len(LABEL_MAP)
                    ][:3]
            
            return {
                "predicted_word": predicted_word,
                "confidence_score": confidence_value,
                "current_sentence": sentence,
                "alternatives": alternatives
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "predicted_word": "",
                "confidence_score": 0,
                "current_sentence": "",
                "alternatives": []
            }
    
    def clear_session(self, session_id: str):
        if session_id in self.frame_sequences:
            del self.frame_sequences[session_id]
        if session_id in self.sentence_buffers:
            del self.sentence_buffers[session_id]
        if session_id in self.last_word_times:
            del self.last_word_times[session_id]
        if session_id in self.last_predictions:
            del self.last_predictions[session_id]

# Global processor
processor = SignLanguageProcessor()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        self.active_connections[session_id] = websocket
        logger.info(f"Client {session_id} connected")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            processor.clear_session(session_id)
            logger.info(f"Client {session_id} disconnected")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to {session_id}: {e}")

manager = ConnectionManager()

@router.websocket("/stream")
async def sign_language_stream(websocket: WebSocket):
    session_id = None
    
    try:
        # Accept connection ONCE
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Wait for session token message
        try:
            init_data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            init_json = json.loads(init_data)
            session_id = init_json.get("session_token", f"session_{id(websocket)}")
            logger.info(f"Session token received: {session_id}")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session token")
            await websocket.close(code=1008, reason="Session token required")
            return
        except Exception as e:
            logger.error(f"Error receiving session token: {e}")
            await websocket.close(code=1008, reason="Invalid session token")
            return
        
        # Register connection
        await manager.connect(websocket, session_id)
        
        # Send confirmation
        await manager.send_json(session_id, {
            "event": "connected",
            "session_id": session_id,
            "message": "Connected to sign language service",
            "model_loaded": model is not None
        })
        
        # Process incoming frames
        frame_count = 0
        last_inference_time = time.time()
        inference_interval = 0.3  # 300ms between inferences
        
        while True:
            try:
                # Receive binary frame data
                frame_data = await websocket.receive_bytes()
                frame_count += 1
                
                # Process frame at limited FPS
                current_time = time.time()
                if current_time - last_inference_time >= inference_interval:
                    result = processor.process_frame(session_id, frame_data)
                    last_inference_time = current_time
                    
                    if result.get("error"):
                        await manager.send_json(session_id, {
                            "event": "error",
                            "error": result["error"]
                        })
                    else:
                        # Send partial text update
                        if "current_sentence" in result and result["current_sentence"]:
                            await manager.send_json(session_id, {
                                "event": "text_streamed",
                                "partial_text": result.get("current_sentence", ""),
                                "confidence_score": result.get("confidence_score", 0),
                                "hand_detected": result.get("hand_detected", False),
                                "sequence_length": result.get("sequence_length", 0)
                            })
                        
                        # Send full inference result
                        if (result.get("predicted_word") and 
                            result.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD):
                            await manager.send_json(session_id, {
                                "event": "inference_complete",
                                "raw_prediction_text": result["predicted_word"],
                                "final_confidence_score": result["confidence_score"],
                                "full_sentence": result.get("current_sentence", ""),
                                "alternatives": result.get("alternatives", [])
                            })
                
            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await manager.send_json(session_id, {
                    "event": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected before session initialization")
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
        "sequence_length": SEQUENCE_LENGTH,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }