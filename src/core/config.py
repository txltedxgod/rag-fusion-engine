from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-fusion-engine"
    app_env: str = "production"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    rrf_k: int = Field(default=60, description="RRF smoothing constant k")
    vector_dim: int = Field(default=128, description="Dense vector dimensions")
    max_expanded_queries: int = Field(default=3, ge=1, le=10)
    log_level: str = "INFO"

settings = Settings()
