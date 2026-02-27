"""link embeddings to uploaded images

Revision ID: 0003_img_embed_link
Revises: 0002_emp_profile_imgs
Create Date: 2026-02-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_img_embed_link"
down_revision = "0002_emp_profile_imgs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employee_face_embeddings", sa.Column("source_image_id", sa.Integer(), nullable=True))
    op.add_column("employee_reid_embeddings", sa.Column("source_image_id", sa.Integer(), nullable=True))

    # Best-effort backfill for existing rows: match image/embedding pairs by (employee_id, created_at order).
    op.execute(
        """
        WITH img AS (
            SELECT id, employee_id,
                   row_number() OVER (PARTITION BY employee_id ORDER BY created_at, id) AS rn
            FROM employee_uploaded_images
            WHERE kind = 'face'
        ),
        emb AS (
            SELECT id, employee_id,
                   row_number() OVER (PARTITION BY employee_id ORDER BY created_at, id) AS rn
            FROM employee_face_embeddings
            WHERE source_image_id IS NULL
        )
        UPDATE employee_face_embeddings e
        SET source_image_id = i.id
        FROM emb b
        JOIN img i
          ON i.employee_id = b.employee_id
         AND i.rn = b.rn
        WHERE e.id = b.id
        """
    )

    op.execute(
        """
        WITH img AS (
            SELECT id, employee_id,
                   row_number() OVER (PARTITION BY employee_id ORDER BY created_at, id) AS rn
            FROM employee_uploaded_images
            WHERE kind = 'reid'
        ),
        emb AS (
            SELECT id, employee_id,
                   row_number() OVER (PARTITION BY employee_id ORDER BY created_at, id) AS rn
            FROM employee_reid_embeddings
            WHERE source_image_id IS NULL
        )
        UPDATE employee_reid_embeddings e
        SET source_image_id = i.id
        FROM emb b
        JOIN img i
          ON i.employee_id = b.employee_id
         AND i.rn = b.rn
        WHERE e.id = b.id
        """
    )

    op.create_index(
        "ix_employee_face_embeddings_source_image_id",
        "employee_face_embeddings",
        ["source_image_id"],
        unique=False,
    )
    op.create_index(
        "ix_employee_reid_embeddings_source_image_id",
        "employee_reid_embeddings",
        ["source_image_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_face_embedding_source_image_id",
        "employee_face_embeddings",
        "employee_uploaded_images",
        ["source_image_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_reid_embedding_source_image_id",
        "employee_reid_embeddings",
        "employee_uploaded_images",
        ["source_image_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_reid_embedding_source_image_id", "employee_reid_embeddings", type_="foreignkey")
    op.drop_constraint("fk_face_embedding_source_image_id", "employee_face_embeddings", type_="foreignkey")

    op.drop_index("ix_employee_reid_embeddings_source_image_id", table_name="employee_reid_embeddings")
    op.drop_index("ix_employee_face_embeddings_source_image_id", table_name="employee_face_embeddings")

    op.drop_column("employee_reid_embeddings", "source_image_id")
    op.drop_column("employee_face_embeddings", "source_image_id")
