# app/models/whisper/loader.py

import whisper
import torch
from pathlib import Path
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()


class WhisperLoader:
    """Singleton loader for Whisper model (base or fine-tuned)."""

    _model: whisper.Whisper | None = None

    @classmethod
    def get_model(cls) -> whisper.Whisper:
        if cls._model is None:
            cls._model = cls._load()
        return cls._model

    @classmethod
    def _load(cls) -> whisper.Whisper:
        settings = get_settings()
        device = settings.whisper_device

        logger.info("Loading Whisper", size=settings.whisper_model_size, device=device)

        if settings.whisper_fine_tuned_path and Path(settings.whisper_fine_tuned_path).exists():
            # Load fine-tuned checkpoint
            logger.info("Using fine-tuned Whisper", path=settings.whisper_fine_tuned_path)
            model = whisper.load_model(settings.whisper_model_size, device=device)
            checkpoint = torch.load(settings.whisper_fine_tuned_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Load base Whisper model
            model = whisper.load_model(settings.whisper_model_size, device=device)

        model.eval()
        logger.info("Whisper loaded successfully")
        return model