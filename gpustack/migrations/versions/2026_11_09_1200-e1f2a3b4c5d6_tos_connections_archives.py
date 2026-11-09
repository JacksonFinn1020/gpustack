"""TOS connections, archive jobs, and provenance columns.

Revision ID: e1f2a3b4c5d6
Revises: d9f1a2b3c4e5
Create Date: 2026-11-09 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d9f1a2b3c4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not table_exists("tos_connections"):
        op.create_table(
            "tos_connections",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("endpoint", sa.String(length=512), nullable=False),
            sa.Column("region", sa.String(length=128), nullable=False),
            sa.Column("bucket", sa.String(length=255), nullable=False),
            sa.Column("prefix", sa.String(length=512), nullable=False, server_default=""),
            sa.Column("credential_mode", sa.String(length=32), nullable=False),
            sa.Column("credential_ref", sa.String(length=255), nullable=True),
            sa.Column("access_key", sa.String(length=255), nullable=True),
            sa.Column("secret_key", sa.Text(), nullable=True),
            sa.Column("security_token", sa.Text(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("owner_principal_id", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint(
                "owner_principal_id",
                "name",
                name="uix_tos_connections_name_per_owner",
            ),
        )

    if not table_exists("tos_archive_jobs"):
        op.create_table(
            "tos_archive_jobs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("model_file_id", sa.Integer(), nullable=False),
            sa.Column("connection_id", sa.Integer(), nullable=False),
            sa.Column("worker_id", sa.Integer(), nullable=False),
            sa.Column("tos_model", sa.String(length=512), nullable=False),
            sa.Column("tos_version", sa.String(length=255), nullable=False),
            sa.Column("paths", sa.JSON(), nullable=True),
            sa.Column("state", sa.String(length=32), nullable=False),
            sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
            sa.Column("size", sa.BigInteger(), nullable=True),
            sa.Column("uploaded_bytes", sa.BigInteger(), nullable=True),
            sa.Column("state_message", sa.Text(), nullable=True),
            sa.Column("error_code", sa.String(length=64), nullable=True),
            sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("provenance", sa.JSON(), nullable=True),
            sa.Column("cluster_id", sa.Integer(), nullable=True),
            sa.Column("owner_principal_id", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["model_file_id"], ["model_files.id"]),
            sa.ForeignKeyConstraint(["connection_id"], ["tos_connections.id"]),
            sa.ForeignKeyConstraint(["worker_id"], ["workers.id"]),
            sa.UniqueConstraint(
                "connection_id",
                "tos_model",
                "tos_version",
                name="uix_tos_archive_connection_model_version",
            ),
        )

    for table in ("models", "model_files", "model_instances"):
        if table_exists(table) and not column_exists(table, "tos_connection_id"):
            op.add_column(
                table, sa.Column("tos_connection_id", sa.Integer(), nullable=True)
            )
        if table_exists(table) and not column_exists(table, "auto_archive"):
            op.add_column(table, sa.Column("auto_archive", sa.JSON(), nullable=True))
        if table_exists(table) and not column_exists(table, "provenance"):
            op.add_column(table, sa.Column("provenance", sa.JSON(), nullable=True))

    if table_exists("model_files") and not column_exists("model_files", "archive_state"):
        op.add_column(
            "model_files",
            sa.Column("archive_state", sa.String(length=32), nullable=True),
        )
    if table_exists("model_files") and not column_exists(
        "model_files", "archive_error_code"
    ):
        op.add_column(
            "model_files",
            sa.Column("archive_error_code", sa.String(length=64), nullable=True),
        )
    if table_exists("model_files") and not column_exists(
        "model_files", "archive_error_message"
    ):
        op.add_column(
            "model_files",
            sa.Column("archive_error_message", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    if table_exists("tos_archive_jobs"):
        op.drop_table("tos_archive_jobs")
    if table_exists("tos_connections"):
        op.drop_table("tos_connections")
    for table in ("models", "model_files", "model_instances"):
        for col in ("tos_connection_id", "auto_archive", "provenance"):
            if table_exists(table) and column_exists(table, col):
                op.drop_column(table, col)
    for col in ("archive_state", "archive_error_code", "archive_error_message"):
        if table_exists("model_files") and column_exists("model_files", col):
            op.drop_column("model_files", col)
