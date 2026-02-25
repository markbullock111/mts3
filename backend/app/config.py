from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/attendance_db",
        alias="DATABASE_URL",
    )
    attendance_timezone: str = Field(default="UTC", alias="ATTENDANCE_TIMEZONE")
    backend_host: str = Field(default="127.0.0.1", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    ui_origins: str = Field(default="http://127.0.0.1:8000,http://localhost:8000", alias="UI_ORIGINS")
    repo_root: Path = Path(__file__).resolve().parents[2]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def ui_origin_list(self) -> List[str]:
        return [x.strip() for x in self.ui_origins.split(",") if x.strip()]


settings = AppSettings()  # type: ignore[call-arg]
