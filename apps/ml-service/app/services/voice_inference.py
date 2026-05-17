import asyncio
import time
import numpy as np
import io
from loguru import logger
import soundfile as sf

from app.schemas.voice import VoiceInferenceResult

class VoiceInferenceService:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            import torch
            import whisper
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = whisper.load_model(self.model_name, device=device)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        session_token: str,
        inference_id: str,
    ) -> VoiceInferenceResult:
        """Transcribe audio chunk using memory buffer (no temp files)"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing audio: {len(audio_bytes)} bytes")
            
            # Convert bytes to numpy array using soundfile
            audio_buffer = io.BytesIO(audio_bytes)
            
            try:
                # Read audio data
                data, samplerate = sf.read(audio_buffer)
                logger.info(f"Audio loaded with soundfile: sample_rate={samplerate}, shape={data.shape}")
                
            except Exception as e:
                logger.error(f"Soundfile failed to read: {e}")
                # Try with wavio as fallback
                try:
                    import wavio
                    wav = wavio.read(audio_buffer)
                    data = wav.data.astype(np.float32) / 32768.0
                    samplerate = wav.rate
                    logger.info(f"Audio loaded with wavio: sample_rate={samplerate}, shape={data.shape}")
                except Exception as e2:
                    logger.error(f"Wavio also failed: {e2}")
                    raise ValueError(f"Cannot read audio file: {e}")
            
            # Ensure mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                logger.info(f"Converted to mono, new shape: {data.shape}")
            
            duration = len(data) / samplerate
            logger.info(f"Audio info: sample_rate={samplerate}, duration={duration:.2f}s, samples={len(data)}")
            
            if duration < 0.5:
                logger.warning(f"Audio too short: {duration:.2f}s")
                return VoiceInferenceResult(
                    raw_prediction_text="",
                    confidence_score=0.0,
                    model_version=self.model_name,
                    inference_latency_ms=int((time.time() - start_time) * 1000),
                    error_message="Audio terlalu pendek (minimal 0.5 detik)"
                )
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if samplerate != 16000:
                logger.info(f"Resampling from {samplerate}Hz to 16000Hz")
                import torch
                import torchaudio
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(data).float()
                
                # Add channel dimension if needed
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Resample
                resampler = torchaudio.transforms.Resample(samplerate, 16000)
                audio_tensor = resampler(audio_tensor)
                
                # Convert back to numpy and remove channel dimension
                data = audio_tensor.squeeze().numpy()
                samplerate = 16000
                logger.info(f"Resampled to {samplerate}Hz, new shape: {data.shape}")
            
            # Normalize audio
            if np.abs(data).max() > 0:
                data = data / np.abs(data).max()
            
            logger.info(f"Running Whisper transcription...")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    data,
                    language="id",
                    task="transcribe",
                    verbose=False,
                    fp16=False,
                    temperature=0.0
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription successful: '{transcribed_text}' (latency: {latency_ms:.0f}ms)")
            
            if not transcribed_text:
                logger.warning("No text detected in audio")
                return VoiceInferenceResult(
                    raw_prediction_text="",
                    confidence_score=0.0,
                    model_version=self.model_name,
                    inference_latency_ms=int(latency_ms),
                    error_message="Tidak ada suara terdeteksi"
                )
            
            return VoiceInferenceResult(
                raw_prediction_text=transcribed_text,
                confidence_score=0.9,
                model_version=self.model_name,
                inference_latency_ms=int(latency_ms),
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return VoiceInferenceResult(
                raw_prediction_text="",
                confidence_score=0.0,
                model_version=self.model_name,
                inference_latency_ms=int(latency_ms),
                error_message=str(e)
            )