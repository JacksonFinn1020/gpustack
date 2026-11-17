"""Add TOS object-storage model source fields.

Adds ``tos`` to ``sourceenum`` where that native type still exists, and
nullable coordinate columns on models, model_instances and model_files.

Revision ID: d9f1a2b3c4e5
Revises: c8d3e6f1a902
Create Date: 2026-11-04 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

import gpustack.utils.sql_enum as sql_enum
from gpustack.migrations.utils import column_exists, table_exists


revision: str = 'd9f1a2b3c4e5'
down_revision: Union[str, None] = 'c8d3e6f1a902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ('models', 'model_instances', 'model_files')
_COLUMNS = (
    'tos_endpoint',
    'tos_region',
    'tos_bucket',
    'tos_model',
    'tos_version',
    'tos_release_path',
)
_SOURCE_TABLE_COLUMNS = {
    'models': 'source',
    'model_instances': 'source',
    'model_files': 'source',
}


def _pg_enum_labels(conn, type_name: str) -> set:
    rows = conn.execute(
        sa.text(
            "SELECT e.enumlabel FROM pg_enum e "
            "JOIN pg_type t ON t.oid = e.enumtypid "
            "WHERE t.typname = :name"
        ),
        {'name': type_name},
    ).fetchall()
    return {row[0] for row in rows}


def _add_source_enum() -> None:
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        labels = _pg_enum_labels(conn, 'sourceenum')
        if not labels:
            return
        # Historical rows use the member name; the API value is ``tos``.
        for value in ('TOS', 'tos'):
            if value not in labels:
                conn.execute(
                    sa.text(f"ALTER TYPE sourceenum ADD VALUE '{value}'")
                )
        return
    if conn.dialect.name == 'mysql':
        definition = conn.execute(
            sa.text(
                "SELECT COLUMN_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_NAME = 'models' AND COLUMN_NAME = 'source' "
                "AND TABLE_SCHEMA = DATABASE()"
            )
        ).first()
        if not definition or not str(definition[0]).lower().startswith('enum('):
            return
        sql_enum.add_enum_values(
            _SOURCE_TABLE_COLUMNS,
            sa.Enum(
                'HUGGING_FACE',
                'MODEL_SCOPE',
                'LOCAL_PATH',
                name='sourceenum',
            ),
            'TOS',
            'tos',
        )


def upgrade() -> None:
    _add_source_enum()
    for table in _TABLES:
        if not table_exists(table):
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            for column in _COLUMNS:
                if not column_exists(table, column):
                    batch_op.add_column(
                        sa.Column(
                            column,
                            sqlmodel.sql.sqltypes.AutoString(),
                            nullable=True,
                        )
                    )


def downgrade() -> None:
    # Enum labels are left in place: removing a PostgreSQL enum value is
    # not reversible without rewriting every column that uses the type.
    for table in _TABLES:
        if not table_exists(table):
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            for column in _COLUMNS:
                if column_exists(table, column):
                    batch_op.drop_column(column)
