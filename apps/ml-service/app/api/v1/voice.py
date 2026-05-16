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


def get_emitter() -> SocketIOEmitter:
    global _emitter
    if _emitter is None:
        _emitter = SocketIOEmitter()
    return _emitter


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
    emitter = get_emitter()

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