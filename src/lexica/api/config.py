from pydantic import BaseModel
import os
from pathlib import Path


class Settings(BaseModel):
    artifacts_dir: str = os.getenv("LEXICA_ARTIFACTS_DIR", "./artifacts")
    cors_origins: list[str] = os.getenv(
        "LEXICA_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")

    # Engine: mock | llm | hybrid
    engine: str = os.getenv("LEXICA_ENGINE", "hybrid").lower()


settings = Settings()
Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)
