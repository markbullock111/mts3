"""initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2026-02-25 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


# Use PostgreSQL ENUM with create_type=False because we create/drop explicitly with checkfirst=True.
# This avoids duplicate CREATE TYPE when Alembic creates tables.
employee_status = postgresql.ENUM("active", "inactive", name="employee_status", create_type=False)
attendance_method = postgresql.ENUM("face", "reid", "unknown", name="attendance_method", create_type=False)


def upgrade() -> None:
    employee_status.create(op.get_bind(), checkfirst=True)
    attendance_method.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "employees",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("employee_code", sa.String(length=64), nullable=False),
        sa.Column("status", employee_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_employees_employee_code", "employees", ["employee_code"], unique=True)

    op.create_table(
        "employee_face_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding_vector", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_employee_face_embeddings_employee_id", "employee_face_embeddings", ["employee_id"], unique=False)

    op.create_table(
        "employee_reid_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding_vector", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_employee_reid_embeddings_employee_id", "employee_reid_embeddings", ["employee_id"], unique=False)

    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("rtsp_url", sa.Text(), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=128), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )

    op.create_table(
        "attendance_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("method", attendance_method, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("camera_id", sa.String(length=128), nullable=False),
        sa.Column("track_uid", sa.String(length=128), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("camera_id", "track_uid", "ts", name="uq_attendance_track_ts"),
    )
    op.create_index("ix_attendance_events_employee_id", "attendance_events", ["employee_id"], unique=False)
    op.create_index("ix_attendance_events_ts", "attendance_events", ["ts"], unique=False)
    op.create_index("ix_attendance_events_camera_id", "attendance_events", ["camera_id"], unique=False)
    op.create_index("ix_attendance_events_track_uid", "attendance_events", ["track_uid"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_attendance_events_track_uid", table_name="attendance_events")
    op.drop_index("ix_attendance_events_camera_id", table_name="attendance_events")
    op.drop_index("ix_attendance_events_ts", table_name="attendance_events")
    op.drop_index("ix_attendance_events_employee_id", table_name="attendance_events")
    op.drop_table("attendance_events")
    op.drop_table("settings")
    op.drop_table("cameras")
    op.drop_index("ix_employee_reid_embeddings_employee_id", table_name="employee_reid_embeddings")
    op.drop_table("employee_reid_embeddings")
    op.drop_index("ix_employee_face_embeddings_employee_id", table_name="employee_face_embeddings")
    op.drop_table("employee_face_embeddings")
    op.drop_index("ix_employees_employee_code", table_name="employees")
    op.drop_table("employees")
    attendance_method.drop(op.get_bind(), checkfirst=True)
    employee_status.drop(op.get_bind(), checkfirst=True)
