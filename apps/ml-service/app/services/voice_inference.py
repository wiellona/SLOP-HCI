# app/services/voice_inference.py

import io
import os
import tempfile
import time
import uuid
from typing import AsyncGenerator

import numpy as np
import structlog
import torch
import whisper

from app.core.config import get_settings
from app.models.whisper.loader import WhisperLoader
from app.schemas.voice import VoiceInferenceResult

logger = structlog.get_logger()


class VoiceInferenceService:
    """Handles speech-to-text inference using Whisper.

    Supports:
      - Single-chunk transcription (short utterances)
      - Streaming mode (chunked audio accumulation)
    """

    SAMPLE_RATE = 16_000  # Whisper selalu mengekspektasi 16kHz mono

    def __init__(self):
        self.model = WhisperLoader.get_model()
        self.settings = get_settings()

    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        session_token: str,
        inference_id: str | None = None,
    ) -> VoiceInferenceResult:
        """Transcribe a single audio chunk (WAV bytes at 16kHz mono).

        Args:
            audio_bytes: Raw WAV audio bytes
            session_token: Conversation session identifier
            inference_id: Optional ID to link streaming events

        Returns:
            VoiceInferenceResult with transcription and confidence
        """
        if inference_id is None:
            inference_id = str(uuid.uuid4())

        start_time = time.monotonic()

        # Decode audio ke numpy float32 array
        audio_array = self._bytes_to_array(audio_bytes)

        # Jalankan Whisper inference
        result = self.model.transcribe(
            audio_array,
            language=self.settings.whisper_language,  # "id" untuk Bahasa Indonesia
            task="transcribe",
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,  # Lebih baik untuk ucapan pendek
            without_timestamps=True,
        )

        latency_ms = int((time.monotonic() - start_time) * 1000)
        text = result["text"].strip()
        confidence = self._extract_confidence(result)

        logger.info(
            "Voice transcription complete",
            session=session_token,
            text_preview=text[:50],
            confidence=confidence,
            latency_ms=latency_ms,
        )

        return VoiceInferenceResult(
            inference_id=inference_id,
            session_token=session_token,
            raw_prediction_text=text,
            confidence_score=confidence,
            model_version=f"whisper-{self.settings.whisper_model_size}",
            inference_latency_ms=latency_ms,
        )

    async def transcribe_streaming(
        self,
        audio_chunks: list[bytes],
        session_token: str,
    ) -> AsyncGenerator[str, None]:
        """Streaming transcription: menghasilkan kata parsial seiring proses

        decoding.
        """
        combined = b"".join(audio_chunks)
        audio_array = self._bytes_to_array(combined)

        result = self.model.transcribe(
            audio_array,
            language=self.settings.whisper_language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            without_timestamps=False,
        )

        accumulated = ""
        for segment in result.get("segments", []):
            segment_text = segment["text"].strip()
            if segment_text:
                accumulated += " " + segment_text
                yield accumulated.strip()

    def _bytes_to_array(self, audio_bytes_or_buf) -> np.ndarray:
        """Mengubah objek audio bytes atau BytesIO menjadi numpy array yang

        valid untuk Whisper.
        """
        # 1. Ambal data bytes mentah baik dari objek BytesIO maupun bytes langsung
        if isinstance(audio_bytes_or_buf, io.BytesIO):
            data = audio_bytes_or_buf.getvalue()
        else:
            data = audio_bytes_or_buf

        # 2. Bikin file temporary fisik di disk agar bisa dibaca oleh ffmpeg bawaan Whisper
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name

        try:
            # 3. Umpankan string path file temporary ke Whisper
            audio_array = whisper.load_audio(tmp_path)
        finally:
            # 4. Hapus kembali file temporary agar tidak menumpuk memenuhi harddisk
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return audio_array

    def _extract_confidence(self, whisper_result: dict) -> float:
        """Mengekstrak confidence proxy dari nilai avg_logprob milik Whisper.

        Memetakan nilai logprob ke dalam rentang [0.0, 1.0].
        """
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.5

        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
        confidence = float(np.exp(np.clip(avg_logprob, -5.0, 0.0)))
        return round(confidence, 4)