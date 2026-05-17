from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import torch
from typing import Optional
import structlog

logger = structlog.get_logger()

router = APIRouter()

# Load Whisper model once at startup
_model = None

def get_model():
    global _model
    if _model is None:
        # Use small model for faster inference, or base for better accuracy
        _model = whisper.load_model("base")
        logger.info("Whisper model loaded")
    return _model

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper.
    
    Accepts: WAV, MP3, M4A, OGG, FLAC formats
    Returns: transcribed text with confidence score
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Load and transcribe
        model = get_model()
        result = model.transcribe(
            tmp_path,
            language="id",  # Bahasa Indonesia
            task="transcribe",
            fp16=torch.cuda.is_available(),
        )
        
        text = result["text"].strip()
        confidence = calculate_confidence(result)
        
        logger.info(
            "Voice transcribed",
            text_preview=text[:50],
            confidence=confidence,
            duration=result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
        )
        
        return JSONResponse({
            "success": True,
            "text": text,
            "confidence": confidence,
            "language": result.get("language", "id"),
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/transcribe/stream")
async def transcribe_streaming(audio: UploadFile = File(...)):
    """
    Streaming transcription endpoint - returns partial results as they become available.
    For true streaming, use WebSocket instead.
    """
    # For now, same as regular transcribe
    return await transcribe_audio(audio)


def calculate_confidence(whisper_result: dict) -> float:
    """Calculate confidence score from Whisper result."""
    segments = whisper_result.get("segments", [])
    if not segments:
        return 0.5
    
    avg_logprob = 0
    total_len = 0
    
    for segment in segments:
        logprob = segment.get("avg_logprob", -1.0)
        length = len(segment.get("text", ""))
        avg_logprob += logprob * length
        total_len += length
    
    if total_len > 0:
        avg_logprob /= total_len
        confidence = float(torch.exp(torch.tensor(avg_logprob)).item())
        return round(min(max(confidence, 0.0), 1.0), 4)
    
    return 0.5