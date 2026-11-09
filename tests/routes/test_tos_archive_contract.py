"""API contract checks that do not need a live database or TOS account."""

import pytest

from gpustack.routes.routes import SYSTEM_ROUTES
from gpustack.routes import tos_archives
from gpustack.schemas.models import ModelSource, SourceEnum
from gpustack.schemas.tos import (
    TosArchiveJobCreate,
    TosConnectionCreate,
    TosConnectionPublic,
    TosTransport,
)


def test_worker_runtime_includes_archive_callbacks():
    endpoints = {item[1] for item in SYSTEM_ROUTES}
    assert tos_archives.list_archives in endpoints
    assert tos_archives.get_archive_transport in endpoints
    assert tos_archives.complete_archive in endpoints
    assert tos_archives.create_archive not in endpoints
    assert tos_archives.retry_archive not in endpoints
    assert tos_archives.list_release_catalog not in endpoints


def test_connection_create_requires_inline_secrets():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TosConnectionCreate(
            name="prod",
            endpoint="https://tos-cn-beijing.volces.com",
            region="cn-beijing",
            bucket="weights",
        )
    TosConnectionCreate(
        name="prod",
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        access_key="AKLTABC",
        secret_key="secret",
    )
    TosConnectionCreate(
        name="env",
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        credential_mode="env",
        credential_ref="env",
    )


def test_connection_public_has_no_secret_fields():
    fields = TosConnectionPublic.model_fields
    assert "secret_key" not in fields
    assert "security_token" not in fields
    assert "access_key" in fields


def test_transport_is_worker_only_shape():
    payload = TosTransport(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        prefix="org-a",
        tos_model="qwen3",
        tos_version="v1",
        credential_mode="inline",
        access_key="AK",
        secret_key="SK",
    )
    assert payload.secret_key == "SK"


def test_archive_create_examples_cover_dir_and_file():
    full = TosArchiveJobCreate(
        model_file_id=1,
        connection_id=2,
        tos_model="qwen3",
        tos_version="v1",
    )
    assert full.paths is None
    single = TosArchiveJobCreate(
        model_file_id=1,
        connection_id=2,
        tos_model="qwen3",
        tos_version="gguf-q4",
        paths=["model.gguf"],
    )
    assert single.paths == ["model.gguf"]


def test_legacy_tos_source_fields_still_valid():
    source = ModelSource(
        source=SourceEnum.TOS,
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
        tos_release_path="lingji-models/qwen3/v1",
    )
    assert source.tos_connection_id is None
    assert "tos_endpoint" in source.source_create_fields()
