from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from loguru import logger
from pydantic import BaseModel

# Create router
router = APIRouter()

# Response model
class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    latency_ms: int
    error: Optional[str] = None

# We'll initialize the service lazily to avoid import issues
_voice_service = None

def get_voice_service():
    global _voice_service
    if _voice_service is None:
        from app.services.voice_inference import VoiceInferenceService
        _voice_service = VoiceInferenceService(model_name="base")
    return _voice_service

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    session_token: Optional[str] = Form(None),
):
    """
    Transcribe audio file to text using Whisper
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Received audio file: {audio.filename}, size: {len(audio_bytes)} bytes, content_type: {audio.content_type}")
        
        # Get service and transcribe
        service = get_voice_service()
        result = await service.transcribe_chunk(
            audio_bytes=audio_bytes,
            session_token=session_token or "unknown",
            inference_id="transcribe"
        )
        
        # Return response even if there's an error in the result
        return TranscriptionResponse(
            text=result.raw_prediction_text,
            confidence=result.confidence_score,
            latency_ms=result.inference_latency_ms,
            error=result.error_message
        )
        
    except Exception as e:
        logger.error(f"Transcription endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def voice_health():
    """Check if voice service is working"""
    try:
        service = get_voice_service()
        return {"status": "healthy", "model": service.model_name}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}