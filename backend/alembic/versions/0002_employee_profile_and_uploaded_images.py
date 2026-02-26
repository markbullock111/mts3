"""employee profile fields and uploaded images

Revision ID: 0002_emp_profile_imgs
Revises: 0001_initial
Create Date: 2026-02-25 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_emp_profile_imgs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employees", sa.Column("birth_date", sa.Date(), nullable=True))
    op.add_column("employees", sa.Column("job_title", sa.String(length=255), nullable=True))
    op.add_column("employees", sa.Column("address", sa.Text(), nullable=True))

    op.create_table(
        "employee_uploaded_images",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_employee_uploaded_images_employee_id", "employee_uploaded_images", ["employee_id"], unique=False)
    op.create_index("ix_employee_uploaded_images_kind", "employee_uploaded_images", ["kind"], unique=False)
    op.create_index("ix_employee_uploaded_images_created_at", "employee_uploaded_images", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_employee_uploaded_images_created_at", table_name="employee_uploaded_images")
    op.drop_index("ix_employee_uploaded_images_kind", table_name="employee_uploaded_images")
    op.drop_index("ix_employee_uploaded_images_employee_id", table_name="employee_uploaded_images")
    op.drop_table("employee_uploaded_images")
    op.drop_column("employees", "address")
    op.drop_column("employees", "job_title")
    op.drop_column("employees", "birth_date")
