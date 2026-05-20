import numpy as np
import torch
from collections import deque
from typing import List, Dict, Any, Optional
from loguru import logger
import time
import os

class SignInferenceService:
    def __init__(self, model_path: str, label_map: List[str], sequence_length: int = 30):
        self.model_path = model_path
        self.label_map = label_map
        self.sequence_length = sequence_length
        self.model = None
        self.device = None
        self.confidence_threshold = 0.7
        self.no_hand_timeout = 3.0
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load Siformer model"""
        try:
            import torch
            
            # Check device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Import Siformer model
            from app.models.siformer_model import Siformer
            
            # Initialize model with correct dimensions for training
            num_classes = len(self.label_map)
            
            # Try different configurations
            configs_to_try = [
                {"dim_feedforward": 2048, "d_model": 128, "nhead": 8, "num_layers": 3},
                {"dim_feedforward": 512, "d_model": 128, "nhead": 8, "num_layers": 3},
                {"dim_feedforward": 2048, "d_model": 256, "nhead": 8, "num_layers": 3},
            ]
            
            model_loaded = False
            
            for config in configs_to_try:
                try:
                    logger.info(f"Trying config: dim_feedforward={config['dim_feedforward']}")
                    
                    self.model = Siformer(
                        in_channels=3,
                        num_joints=21,
                        num_frames=self.sequence_length,
                        num_classes=num_classes,
                        d_model=config['d_model'],
                        nhead=config['nhead'],
                        num_layers=config['num_layers'],
                        dim_feedforward=config['dim_feedforward']
                    )
                    
                    # Load weights
                    if os.path.exists(self.model_path):
                        checkpoint = torch.load(self.model_path, map_location='cpu')
                        
                        # Handle different checkpoint formats
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # Remove 'module.' prefix if present
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        
                        # Filter out mismatched keys
                        model_dict = self.model.state_dict()
                        filtered_dict = {k: v for k, v in state_dict.items() 
                                    if k in model_dict and v.shape == model_dict[k].shape}
                        
                        if len(filtered_dict) > 0:
                            model_dict.update(filtered_dict)
                            self.model.load_state_dict(model_dict, strict=False)
                            self.model = self.model.to(self.device)
                            self.model.eval()
                            model_loaded = True
                            logger.info(f"Model loaded successfully with config: dim_feedforward={config['dim_feedforward']}")
                            logger.info(f"Loaded {len(filtered_dict)}/{len(model_dict)} layers")
                            break
                        else:
                            logger.warning(f"No matching layers found for config {config}")
                    else:
                        logger.warning(f"Model weights not found at {self.model_path}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed with config {config}: {e}")
                    continue
            
            if not model_loaded:
                logger.warning("Could not load real model, falling back to dummy mode")
                self.model = None
                self.dummy_mode = True
            else:
                self.dummy_mode = False
                
        except Exception as e:
            logger.error(f"Failed to load sign model: {e}")
            self.model = None
            self.dummy_mode = True

    def extract_hand_features(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract hand features from MediaPipe landmarks
        Expected input: 21 landmarks with x, y, z coordinates
        """
        features = np.zeros(63)  # 21 landmarks * 3 coordinates
        
        if not landmarks or len(landmarks) < 21:
            return features
        
        # Normalize relative to wrist (landmark 0)
        wrist_x = landmarks[0].get('x', 0)
        wrist_y = landmarks[0].get('y', 0)
        wrist_z = landmarks[0].get('z', 0)
        
        for i, landmark in enumerate(landmarks[:21]):
            if i >= 21:
                break
                
            # Get coordinates
            x = landmark.get('x', 0)
            y = landmark.get('y', 0)
            z = landmark.get('z', 0)
            
            # Normalize
            features[i*3] = x - wrist_x
            features[(i*3) + 1] = y - wrist_y
            features[(i*3) + 2] = z - wrist_z
        
        return features
    
    def predict(self, frame_sequence: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Predict sign language from sequence of frames
        """
        if len(frame_sequence) != self.sequence_length:
            return None
        
        if self.model is None:
            return None
        
        try:
            import torch
            
            # Convert to tensor
            sequence_array = np.array(frame_sequence)  # Shape: (30, 63)
            input_tensor = torch.tensor(sequence_array, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, 30, 63)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                
                confidence_value = confidence.item()
                predicted_idx = predicted_idx.item()
            
            if predicted_idx < len(self.label_map) and confidence_value >= self.confidence_threshold:
                predicted_word = self.label_map[predicted_idx]
                
                return {
                    "word": predicted_word,
                    "confidence": confidence_value,
                    "index": predicted_idx
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if service is ready for inference"""
        return self.model is not None