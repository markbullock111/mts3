from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EmployeeCreate(BaseModel):
    full_name: str
    employee_code: str
    birth_date: date | None = None
    job_title: str | None = None
    address: str | None = None
    status: Literal["active", "inactive"] = "active"


class EmployeeUpdate(BaseModel):
    full_name: str | None = None
    birth_date: date | None = None
    job_title: str | None = None
    address: str | None = None
    status: Literal["active", "inactive"] | None = None
    main_photo_id: int | None = None


class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    employee_code: str
    birth_date: date | None = None
    job_title: str | None = None
    address: str | None = None
    status: str
    main_photo_id: int | None = None
    created_at: datetime
    face_embeddings_count: int = 0
    reid_embeddings_count: int = 0


class EventCreate(BaseModel):
    employee_id: int | None = None
    ts: datetime
    method: Literal["face", "reid", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str
    track_uid: str
    image_path: str | None = None
    meta: dict[str, Any] | None = None


class EventOverride(BaseModel):
    employee_id: int | None


class EventOut(BaseModel):
    id: int
    employee_id: int | None
    ts: datetime
    method: str
    confidence: float
    camera_id: str
    track_uid: str
    image_path: str | None
    employee_name: str | None = None
    employee_code: str | None = None


class SettingsUpdate(BaseModel):
    values: dict[str, Any]


class SettingsOut(BaseModel):
    values: dict[str, Any]


class GalleryPayload(BaseModel):
    version: int
    face_embeddings: list[dict[str, Any]]
    reid_embeddings: list[dict[str, Any]]
