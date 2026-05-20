from pydantic import BaseModel
from typing import Optional

class VoiceInferenceResult(BaseModel):
    raw_prediction_text: str
    confidence_score: float
    model_version: str
    inference_latency_ms: int
    error_message: Optional[str] = None
    # Make these optional with defaults
    inference_id: Optional[str] = None
    session_token: Optional[str] = None
    
    class Config:
        extra = "ignore"