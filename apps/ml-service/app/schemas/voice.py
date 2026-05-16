# app/schemas/voice.py
from pydantic import BaseModel

class VoiceInferenceResult(BaseModel):
    inference_id: str
    session_token: str
    raw_prediction_text: str
    confidence_score: float
    model_version: str
    inference_latency_ms: int