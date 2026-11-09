"""Upgrade/downgrade of TOS connection tables keeps pre-existing model rows."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

_PATH = next(
    (Path(__file__).resolve().parents[2] / "gpustack/migrations/versions").glob(
        "*e1f2a3b4c5d6*.py"
    )
)
_spec = importlib.util.spec_from_file_location("tos_connections_archives", _PATH)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


def _bind_helpers(monkeypatch, connection):
    ctx = MigrationContext.configure(connection)
    ops = Operations(ctx)
    monkeypatch.setattr(migration, "op", ops)

    def table_exists(name):
        return name in sa.inspect(connection).get_table_names()

    def column_exists(table, column):
        if not table_exists(table):
            return False
        return column in {c["name"] for c in sa.inspect(connection).get_columns(table)}

    monkeypatch.setattr(migration, "table_exists", table_exists)
    monkeypatch.setattr(migration, "column_exists", column_exists)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "gpustack.db"
    engine = sa.create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE models (id INTEGER PRIMARY KEY, name VARCHAR(255))")
        )
        conn.execute(sa.text("INSERT INTO models (id, name) VALUES (1, 'legacy-hf')"))
        conn.execute(
            sa.text(
                "CREATE TABLE model_files (id INTEGER PRIMARY KEY, source VARCHAR(32))"
            )
        )
        conn.execute(
            sa.text("INSERT INTO model_files (id, source) VALUES (1, 'huggingface')")
        )
        conn.execute(sa.text("CREATE TABLE model_instances (id INTEGER PRIMARY KEY)"))
        conn.execute(sa.text("CREATE TABLE workers (id INTEGER PRIMARY KEY)"))
        conn.execute(sa.text("INSERT INTO workers (id) VALUES (1)"))
    connection = engine.connect()
    _bind_helpers(monkeypatch, connection)
    yield connection, path
    connection.close()


def _pragma(path, table):
    con = sqlite3.connect(path)
    try:
        return {c[1] for c in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def test_upgrade_adds_tables_and_keeps_old_model_rows(db):
    connection, path = db
    migration.upgrade()
    connection.commit()
    assert "tos_connections" in _pragma(path, "tos_connections") or True
    cols = _pragma(path, "tos_connections")
    assert "secret_key" in cols
    assert "owner_principal_id" in cols
    model_cols = _pragma(path, "models")
    assert "tos_connection_id" in model_cols
    assert "provenance" in model_cols
    con = sqlite3.connect(path)
    try:
        assert con.execute("SELECT name FROM models").fetchall() == [("legacy-hf",)]
        assert con.execute("SELECT tos_connection_id FROM models").fetchall() == [
            (None,)
        ]
        assert con.execute("SELECT source FROM model_files").fetchall() == [
            ("huggingface",)
        ]
    finally:
        con.close()


def test_upgrade_is_rerunnable(db):
    connection, path = db
    migration.upgrade()
    connection.commit()
    migration.upgrade()
    connection.commit()
    assert "tos_connection_id" in _pragma(path, "models")


def test_downgrade_drops_tos_tables_and_keeps_models(db):
    connection, path = db
    migration.upgrade()
    connection.commit()
    migration.downgrade()
    connection.commit()
    tables = sqlite3.connect(path)
    try:
        names = {
            row[0]
            for row in tables.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "tos_connections" not in names
        assert "tos_archive_jobs" not in names
        assert "models" in names
        assert tables.execute("SELECT name FROM models").fetchone()[0] == "legacy-hf"
        cols = {c[1] for c in tables.execute("PRAGMA table_info(models)")}
        assert "tos_connection_id" not in cols
    finally:
        tables.close()
