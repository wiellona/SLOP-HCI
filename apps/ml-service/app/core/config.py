# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    node_server_url: str = "http://localhost:4000"
    ml_service_api_key: str = "duh_apa_ya_xavier_ganteng_deh_1x1x1x" 
    whisper_model_size: str = "small"
    whisper_language: str = "id"
    whisper_device: str = "cpu" 
    whisper_fine_tuned_path: str | None = None
    sign_model_path: str = "app/models/weights/siformer_wlasl_cafe.pth"
    sign_labels_path: str = "app/models/weights/siformer_labels.json"
    sign_translation_map_path: str = "app/resources/sign_translation_id.json"
    sign_use_two_hands: bool = True
    sign_require_both_hands: bool = False
    sign_sequence_length: int = 30
    sign_confidence_threshold: float = 0.7
    sign_inference_interval_ms: int = 120
    sign_min_detection_confidence: float = 0.5
    sign_min_tracking_confidence: float = 0.5
    sign_device: str | None = None

def get_settings():
    return Settings()