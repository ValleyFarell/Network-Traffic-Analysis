"""Настройки приложения, загружаемые из переменных окружения и файла .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Общие настройки скриптов обучения и API."""

    app_name: str = "NAD Host Similarity"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    model_artifact_path: Path = Path("artifacts/similarity_model.pkl")

    database_url: str = "postgresql+psycopg://nad:nad@localhost:5432/nad"

    default_neighbors: int = Field(default=10, ge=1)
    max_neighbors: int = Field(default=100, ge=1)
    model_version: str = "v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NAD_",
        extra="ignore",
    )

    @field_validator("data_dir", "artifacts_dir", "model_artifact_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        """Разворачивает пути вида ~/nad-data из переменных окружения."""

        return Path(value).expanduser()

    @field_validator("max_neighbors")
    @classmethod
    def max_neighbors_must_cover_default(cls, value: int, info) -> int:
        default_neighbors = info.data.get("default_neighbors")
        if default_neighbors is not None and value < default_neighbors:
            raise ValueError("max_neighbors должен быть не меньше default_neighbors")
        return value


@lru_cache
def get_settings() -> Settings:
    """Создаёт настройки один раз и повторно использует их в процессе."""

    return Settings()
