# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    node_server_url: str = "http://localhost:4000"
    ml_service_api_key: str = "duh_apa_ya_xavier_ganteng_deh_1x1x1x" 
    whisper_model_size: str = "small"
    whisper_language: str = "id"
    whisper_device: str = "cpu" 
    whisper_fine_tuned_path: str | None = None

def get_settings():
    return Settings()