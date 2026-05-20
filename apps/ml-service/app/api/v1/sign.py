from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import json
import numpy as np
import cv2
import time
from loguru import logger
from collections import deque
import torch
import mediapipe as mp
import asyncio
import base64
from io import BytesIO
from PIL import Image

router = APIRouter()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
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
        self.frame_counters: Dict[str, int] = {}
        self.last_bounding_boxes: Dict[str, Optional[dict]] = {}
        
    def get_or_create_sequence(self, session_id: str) -> deque:
        if session_id not in self.frame_sequences:
            self.frame_sequences[session_id] = deque(maxlen=SEQUENCE_LENGTH)
            self.sentence_buffers[session_id] = []
            self.last_word_times[session_id] = time.time()
            self.last_predictions[session_id] = ""
            self.frame_counters[session_id] = 0
            self.last_bounding_boxes[session_id] = None
        return self.frame_sequences[session_id]
    
    def decode_frame(self, frame_bytes: bytes):
        """Decode frame from bytes"""
        try:
            # Try to decode as numpy array first
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # If failed, try as PIL image
            if frame is None:
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
            # Decode frame
            frame = self.decode_frame(frame_bytes)
            if frame is None:
                return {"error": "Failed to decode frame"}
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Extract hand features and bounding box
            hand_detected = results.right_hand_landmarks is not None
            bounding_box = None
            hand_features = np.zeros(63)
            
            if hand_detected:
                hand_features = self.extract_hand_features(results.right_hand_landmarks)
                bounding_box = self.calculate_bounding_box(results.right_hand_landmarks, frame.shape)
                
                # Also check left hand if needed
                if results.left_hand_landmarks:
                    # You can combine both hands here if needed
                    pass
            
            # Update sequence
            sequence = self.get_or_create_sequence(session_id)
            sequence.append(hand_features)
            self.frame_counters[session_id] += 1
            
            result = {
                "hand_detected": hand_detected,
                "sequence_length": len(sequence),
                "bounding_box": bounding_box,
                "frame_number": self.frame_counters[session_id]
            }
            
            # Run inference if sequence is full
            if len(sequence) == SEQUENCE_LENGTH:
                inference_result = self.run_inference(sequence, session_id)
                result.update(inference_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return {"error": str(e)}
    
    def run_inference(self, sequence: deque, session_id: str) -> dict:
        try:
            # Convert sequence to tensor
            sequence_matrix = np.array(sequence, dtype=np.float32)
            input_tensor = torch.tensor(sequence_matrix, dtype=torch.float32).unsqueeze(0)
            
            if model is not None:
                with torch.no_grad():
                    output_logits = model(input_tensor)
                    probabilities = torch.softmax(output_logits, dim=1)
                    confidence, predicted_index = torch.max(probabilities, dim=1)
                    confidence_value = confidence.item()
                    predicted_index = predicted_index.item()
            else:
                # Fallback for testing without model
                confidence_value = 0.0
                predicted_index = -1
            
            if predicted_index >= 0 and predicted_index < len(LABEL_MAP):
                predicted_word = LABEL_MAP[predicted_index]
            else:
                predicted_word = ""
                confidence_value = 0.0
            
            # Update sentence buffer
            current_time = time.time()
            last_time = self.last_word_times.get(session_id, current_time)
            last_pred = self.last_predictions.get(session_id, "")
            
            if confidence_value >= CONFIDENCE_THRESHOLD and predicted_word:
                time_diff = current_time - last_time
                
                # Add to sentence buffer if different word or enough time passed
                if predicted_word != last_pred or time_diff > self.sentence_gap_ms / 1000:
                    if time_diff > self.sentence_gap_ms / 1000:
                        # New sentence
                        self.sentence_buffers[session_id] = [predicted_word]
                    else:
                        # Continue current sentence
                        if (not self.sentence_buffers[session_id] or 
                            self.sentence_buffers[session_id][-1] != predicted_word):
                            self.sentence_buffers[session_id].append(predicted_word)
                    
                    self.last_word_times[session_id] = current_time
                    self.last_predictions[session_id] = predicted_word
            
            # Build current sentence
            sentence = " ".join(self.sentence_buffers.get(session_id, []))
            
            # Get alternative predictions
            alternatives = []
            if model is not None and confidence_value < 0.85 and confidence_value > 0.3:
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
            logger.error(f"Inference error: {e}", exc_info=True)
            return {
                "predicted_word": "",
                "confidence_score": 0,
                "current_sentence": "",
                "alternatives": []
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
            "labels": LABEL_MAP,
            "sequence_length": SEQUENCE_LENGTH
        })
        
        logger.info(f"Session {session_id} ready, starting frame processing")
        
        # Frame processing variables
        frame_count = 0
        last_inference_time = time.time()
        inference_interval = 0.2
        last_bounding_box = None
        
        while True:
            try:
                message = await websocket.receive()
                
                if "bytes" in message:
                    # Process binary frame data
                    frame_data = message["bytes"]
                    frame_count += 1
                    
                    # Process frame at limited FPS
                    current_time = time.time()
                    if current_time - last_inference_time >= inference_interval:
                        result = processor.process_frame(session_id, frame_data)
                        last_inference_time = current_time
                        
                        if result.get("error"):
                            logger.warning(f"Frame processing error: {result['error']}")
                            await manager.send_json(session_id, {
                                "event": "error",
                                "error": result["error"]
                            })
                        else:
                            # Send bounding box update
                            bounding_box = result.get("bounding_box")
                            if bounding_box and bounding_box != last_bounding_box:
                                last_bounding_box = bounding_box
                                await manager.send_json(session_id, {
                                    "event": "bounding_box_update",
                                    "bounding_box": bounding_box,
                                    "hand_detected": result.get("hand_detected", False)
                                })
                            
                            # Send text stream update (partial)
                            if result.get("current_sentence"):
                                await manager.send_json(session_id, {
                                    "event": "text_streamed",
                                    "partial_text": result["current_sentence"],
                                    "confidence_score": result.get("confidence_score", 0),
                                    "hand_detected": result.get("hand_detected", False),
                                    "sequence_length": result.get("sequence_length", 0),
                                    "bounding_box": bounding_box
                                })
                            
                            # Send complete inference result
                            if (result.get("predicted_word") and 
                                result.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD):
                                await manager.send_json(session_id, {
                                    "event": "inference_complete",
                                    "raw_prediction_text": result["predicted_word"],
                                    "final_confidence_score": result["confidence_score"],
                                    "full_sentence": result.get("current_sentence", ""),
                                    "alternatives": result.get("alternatives", []),
                                    "bounding_box": bounding_box
                                })
                    
                    if frame_count % 50 == 0:
                        await manager.send_json(session_id, {
                            "event": "heartbeat",
                            "frame_count": frame_count,
                            "timestamp": time.time()
                        })
                        
                elif "text" in message:
                    try:
                        text_data = json.loads(message["text"])
                        logger.info(f"Received text message from {session_id}: {text_data}")
                        
                        if text_data.get("type") == "ping":
                            await manager.send_json(session_id, {
                                "event": "pong",
                                "timestamp": time.time()
                            })
                        elif text_data.get("type") == "reset":
                            processor.clear_session(session_id)
                            await manager.send_json(session_id, {
                                "event": "session_reset",
                                "message": "Session has been reset"
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid text message: {message['text']}")
                
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
        "sequence_length": SEQUENCE_LENGTH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "active_connections": len(manager.active_connections)
    }