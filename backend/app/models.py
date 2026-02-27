from __future__ import annotations

import enum
from datetime import date, datetime, timezone

from sqlalchemy import (
    ARRAY,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmployeeStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class AttendanceMethod(str, enum.Enum):
    face = "face"
    reid = "reid"
    unknown = "unknown"


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    employee_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    job_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[EmployeeStatus] = mapped_column(
        Enum(EmployeeStatus, name="employee_status"), default=EmployeeStatus.active, nullable=False
    )
    main_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_uploaded_images.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    face_embeddings: Mapped[list[EmployeeFaceEmbedding]] = relationship(back_populates="employee", cascade="all, delete-orphan")
    reid_embeddings: Mapped[list[EmployeeReIDEmbedding]] = relationship(back_populates="employee", cascade="all, delete-orphan")
    uploaded_images: Mapped[list[EmployeeUploadedImage]] = relationship(
        back_populates="employee",
        cascade="all, delete-orphan",
        foreign_keys="EmployeeUploadedImage.employee_id",
    )
    main_photo: Mapped[EmployeeUploadedImage | None] = relationship(
        foreign_keys=[main_photo_id],
        post_update=True,
    )
    attendance_events: Mapped[list[AttendanceEvent]] = relationship(back_populates="employee")


class EmployeeFaceEmbedding(Base):
    __tablename__ = "employee_face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), index=True, nullable=False)
    source_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_uploaded_images.id", ondelete="CASCADE"), index=True, nullable=True
    )
    embedding_vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    employee: Mapped[Employee] = relationship(back_populates="face_embeddings")
    source_image: Mapped[EmployeeUploadedImage | None] = relationship(foreign_keys=[source_image_id])


class EmployeeReIDEmbedding(Base):
    __tablename__ = "employee_reid_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), index=True, nullable=False)
    source_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_uploaded_images.id", ondelete="CASCADE"), index=True, nullable=True
    )
    embedding_vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    employee: Mapped[Employee] = relationship(back_populates="reid_embeddings")
    source_image: Mapped[EmployeeUploadedImage | None] = relationship(foreign_keys=[source_image_id])


class EmployeeUploadedImage(Base):
    __tablename__ = "employee_uploaded_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # face|reid
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)  # path relative to data/
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    employee: Mapped[Employee] = relationship(
        back_populates="uploaded_images",
        foreign_keys=[employee_id],
    )


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id", ondelete="SET NULL"), index=True, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    method: Mapped[AttendanceMethod] = mapped_column(Enum(AttendanceMethod, name="attendance_method"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    camera_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    track_uid: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    employee: Mapped[Employee | None] = relationship(back_populates="attendance_events")

    __table_args__ = (
        UniqueConstraint("camera_id", "track_uid", "ts", name="uq_attendance_track_ts"),
    )


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
