"""add temporal fields to monthly_records

Revision ID: 20260511_0002
Revises: 20260511_0001
Create Date: 2026-05-11

"""

from alembic import op
import sqlalchemy as sa


revision = "20260511_0002"
down_revision = "20260511_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("monthly_records", sa.Column("height", sa.Float(), nullable=True))
    op.add_column("monthly_records", sa.Column("smear_tb_lamp", sa.Integer(), nullable=True))
    op.add_column("monthly_records", sa.Column("xpert_mtb_rif", sa.Integer(), nullable=True))
    op.add_column("monthly_records", sa.Column("monthly_doses_taken", sa.Integer(), nullable=True))
    op.add_column("monthly_records", sa.Column("monthly_missed_doses", sa.Integer(), nullable=True))
    op.add_column("monthly_records", sa.Column("cumulative_doses_taken", sa.Integer(), nullable=True))
    op.add_column("monthly_records", sa.Column("pct_adherence", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("monthly_records", "pct_adherence")
    op.drop_column("monthly_records", "cumulative_doses_taken")
    op.drop_column("monthly_records", "monthly_missed_doses")
    op.drop_column("monthly_records", "monthly_doses_taken")
    op.drop_column("monthly_records", "xpert_mtb_rif")
    op.drop_column("monthly_records", "smear_tb_lamp")
    op.drop_column("monthly_records", "height")
