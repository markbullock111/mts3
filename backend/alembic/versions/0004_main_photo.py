"""add employee main photo reference

Revision ID: 0004_main_photo
Revises: 0003_img_embed_link
Create Date: 2026-02-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_main_photo"
down_revision = "0003_img_embed_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employees", sa.Column("main_photo_id", sa.Integer(), nullable=True))
    op.create_index("ix_employees_main_photo_id", "employees", ["main_photo_id"], unique=False)
    op.create_foreign_key(
        "fk_employees_main_photo_id",
        "employees",
        "employee_uploaded_images",
        ["main_photo_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_employees_main_photo_id", "employees", type_="foreignkey")
    op.drop_index("ix_employees_main_photo_id", table_name="employees")
    op.drop_column("employees", "main_photo_id")
