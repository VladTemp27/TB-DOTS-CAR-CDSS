"""v1 schema

Revision ID: 20260511_0001
Revises: 
Create Date: 2026-05-11

"""

from alembic import op
import sqlalchemy as sa


revision = "20260511_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("medical_id", sa.String(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("treatment_regimen", sa.String(), nullable=True),
        sa.Column("treatment_start_date", sa.String(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("failure_probability", sa.Float(), nullable=False),
        sa.Column("contributions", sa.JSON(), nullable=False),
        sa.Column("features_used", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.Integer(), nullable=False),
    )
    op.create_index("ix_predictions_patient_id", "predictions", ["patient_id"])

    op.create_table(
        "monthly_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("month", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("smear_result", sa.String(), nullable=True),
        sa.Column("adherence", sa.String(), nullable=False),
        sa.Column("failure_probability", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.Integer(), nullable=False),
    )
    op.create_index("ix_monthly_records_patient_id", "monthly_records", ["patient_id"])
    op.create_index(
        "ux_monthly_records_patient_month",
        "monthly_records",
        ["patient_id", "month"],
        unique=True,
    )

    op.create_table(
        "xrays",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("month", sa.Integer(), nullable=True),
        sa.Column("mime", sa.String(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("rel_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
    )
    op.create_index("ix_xrays_patient_id", "xrays", ["patient_id"])


def downgrade() -> None:
    op.drop_index("ix_xrays_patient_id", table_name="xrays")
    op.drop_table("xrays")

    op.drop_index("ux_monthly_records_patient_month", table_name="monthly_records")
    op.drop_index("ix_monthly_records_patient_id", table_name="monthly_records")
    op.drop_table("monthly_records")

    op.drop_index("ix_predictions_patient_id", table_name="predictions")
    op.drop_table("predictions")

    op.drop_table("patients")
