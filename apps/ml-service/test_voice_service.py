import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.info("Attempting to import voice_inference...")
    from app.services.voice_inference import VoiceInferenceService
    logger.info("Successfully imported VoiceInferenceService")
    
    logger.info("Attempting to create VoiceInferenceService instance...")
    service = VoiceInferenceService()
    logger.info("Successfully created VoiceInferenceService")
    
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    sys.exit(1)