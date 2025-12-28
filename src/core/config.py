"""
Configuration Management for QuantBet.

Uses pydantic-settings for robust environment variable loading and validation.
"""

from typing import Optional
from pydantic import PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings.

    Loads configuration from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # App Info
    APP_NAME: str = "QuantBet NBL-Quant-Alpha"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Database
    POSTGRES_USER: str = "quantbet"
    POSTGRES_PASSWORD: str = "quantbet_password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "quantbet"

    # Optional override for full URL
    DATABASE_URL: Optional[str] = None

    @computed_field
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """Constructs the Async PostgreSQL DSN."""
        if self.DATABASE_URL:
             # Replace postgresql:// with postgresql+asyncpg:// if needed
            if self.DATABASE_URL.startswith("postgresql://"):
                return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.DATABASE_URL

        return str(PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB
        ))

    # Security
    QUANTBET_API_KEY: Optional[str] = None
    ADMIN_USER: str = "admin"
    ADMIN_PASS: Optional[str] = None

    # External APIs
    ODDS_API_KEY: Optional[str] = None


settings = Settings()
