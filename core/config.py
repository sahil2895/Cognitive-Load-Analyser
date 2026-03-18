import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Cognitive Load Optimization Engine"
    ENVIRONMENT: str = "development"
    
    # AI Models
    GEMINI_API_KEY: str = ""
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    SBERT_MODEL: str = "all-MiniLM-L6-v2"
    
    # ML & Heuristic Weights
    W_INTR: float = 0.45
    W_EXTR: float = 0.45
    W_GERM: float = 0.35
    
    # File Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR: str = os.path.join(BASE_DIR, "models", "saved_models")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
