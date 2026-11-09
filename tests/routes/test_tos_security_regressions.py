"""Negative regressions for TOS tenant isolation, credentials, and complete."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
    InvalidException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import tos_archives
from gpustack.routes.models import _reject_tos_gguf_on_transformer_backend
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import BackendEnum, Model, SourceEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.tos import (
    TosArchiveJob,
    TosArchiveJobUpdate,
    TosArchiveStateEnum,
    TosConnection,
    TosConnectionCreate,
)
from gpustack.server import controllers
from gpustack.server.controllers import _maybe_enqueue_auto_archive
from gpustack.server.tos_bindings import (
    stamp_tos_provenance_from_manifest,
    verify_committed_manifest,
)
from gpustack.utils.tos_secrets import seal_secret
from gpustack.worker import tos_storage
from tests.worker.test_tos_storage import (
    FakeTos,
    _file_entry,
    _location,
    _manifest,
    _sha,
)


def _org_connection(
    owner: int, conn_id: int, secret: str, monkeypatch
) -> TosConnection:
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "unit-test-tos-key")
    return TosConnection(
        id=conn_id,
        name=f"org-{owner}",
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        prefix=f"org-{owner}",
        credential_mode="inline",
        access_key=seal_secret(f"AK-{owner}"),
        secret_key=seal_secret(secret),
        owner_principal_id=owner,
        deleted_at=None,
    )


def _worker_ctx(worker_id: int) -> TenantContext:
    return TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=worker_id)
        ),
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=None,
        scoped_cluster_owner_id=None,
    )


class _SessionCM:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_auto_archive_does_not_enqueue_other_org_connection(monkeypatch):
    file = ModelFile(
        id=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-8B",
        state=ModelFileStateEnum.READY,
        worker_id=7,
        owner_principal_id=10,
        cluster_id=3,
        resolved_paths=["/cache/qwen"],
        auto_archive={
            "enabled": True,
            "connection_id": 2,
            "tos_model": "qwen3",
            "tos_version": "v1",
        },
    )
    conn_b = _org_connection(99, 2, "SK-ORG-B", monkeypatch)
    session = AsyncMock()
    session.commit = AsyncMock()

    async def _update(self, *args, **kwargs):
        return self

    monkeypatch.setattr(ModelFile, "update", _update)
    monkeypatch.setattr(controllers, "async_session", lambda: _SessionCM(session))
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=file))
    monkeypatch.setattr(TosConnection, "one_by_id", AsyncMock(return_value=conn_b))
    enqueue = AsyncMock()
    monkeypatch.setattr("gpustack.server.tos_bindings.enqueue_archive_job", enqueue)

    await _maybe_enqueue_auto_archive(file)

    enqueue.assert_not_called()
    assert file.archive_state == TosArchiveStateEnum.ERROR.value
    assert file.archive_error_code == "connection_not_found"
    assert "SK-ORG-B" not in (file.archive_error_message or "")


@pytest.mark.asyncio
async def test_worker_transport_refuses_other_org_connection_keys(monkeypatch):
    job = TosArchiveJob(
        id=5,
        model_file_id=1,
        connection_id=2,
        worker_id=7,
        tos_model="qwen3",
        tos_version="v1",
        state=TosArchiveStateEnum.UPLOADING.value,
        owner_principal_id=10,
        cluster_id=3,
    )
    conn_b = _org_connection(99, 2, "SK-ORG-B", monkeypatch)
    monkeypatch.setattr(TosArchiveJob, "one_by_id", AsyncMock(return_value=job))
    monkeypatch.setattr(TosConnection, "one_by_id", AsyncMock(return_value=conn_b))
    assigned = _worker_ctx(7)
    other_worker = _worker_ctx(8)
    session = AsyncMock()

    with pytest.raises(ForbiddenException) as other_exc:
        await tos_archives.get_archive_transport(session, other_worker, 5)
    assert "assigned to another worker" in other_exc.value.message
    with pytest.raises(ForbiddenException) as owner_exc:
        await tos_archives.get_archive_transport(session, assigned, 5)
    assert "not owned" in owner_exc.value.message


@pytest.mark.asyncio
async def test_pull_transport_refuses_other_org_and_other_worker(monkeypatch):
    model_file = ModelFile(
        id=11,
        source=SourceEnum.TOS,
        tos_connection_id=2,
        tos_model="qwen3",
        tos_version="v1",
        tos_endpoint="https://tos-cn-beijing.volces.com",
        tos_region="cn-beijing",
        tos_bucket="weights",
        worker_id=7,
        owner_principal_id=10,
        cluster_id=3,
    )
    conn_b = _org_connection(99, 2, "SK-ORG-B", monkeypatch)
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=model_file))
    monkeypatch.setattr(TosConnection, "one_by_id", AsyncMock(return_value=conn_b))
    session = AsyncMock()

    with pytest.raises(ForbiddenException) as other_exc:
        await tos_archives.get_pull_transport(session, _worker_ctx(8), 11)
    assert "assigned to another worker" in other_exc.value.message
    with pytest.raises(ForbiddenException) as owner_exc:
        await tos_archives.get_pull_transport(session, _worker_ctx(7), 11)
    assert "not owned" in owner_exc.value.message


def test_empty_prefix_requires_explicit_bucket_root():
    with pytest.raises(ValidationError, match="allow_bucket_root"):
        TosConnectionCreate(
            name="root",
            endpoint="https://tos-cn-beijing.volces.com",
            region="cn-beijing",
            bucket="weights",
            prefix="",
            access_key="AK",
            secret_key="SK",
        )
    created = TosConnectionCreate(
        name="root",
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        prefix="",
        allow_bucket_root=True,
        access_key="AK",
        secret_key="SK",
    )
    assert created.prefix == ""


def test_put_archive_update_cannot_carry_provenance():
    parsed = TosArchiveJobUpdate.model_validate(
        {"progress": 10, "provenance": {"origin_source": "poison"}}
    )
    dumped = parsed.model_dump()
    assert "provenance" not in dumped
    assert parsed.progress == 10


def test_complete_rejects_missing_remote_manifest(monkeypatch):
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "unit-test-tos-key")
    conn = _org_connection(10, 1, "SK-A", monkeypatch)
    job = TosArchiveJob(
        id=9,
        model_file_id=1,
        connection_id=1,
        worker_id=7,
        tos_model="qwen3",
        tos_version="v1",
        state=TosArchiveStateEnum.VERIFYING.value,
        owner_principal_id=10,
        provenance={
            "origin_source": "huggingface",
            "origin_huggingface_repo_id": "Qwen/Qwen3",
        },
    )
    monkeypatch.setattr(
        "gpustack.server.tos_bindings.client_from_access",
        lambda access: FakeTos({}),
    )
    with pytest.raises(InvalidException) as exc:
        verify_committed_manifest(
            conn, job, {"manifest_sha256": "a" * 64, "origin_source": "poison"}
        )
    assert "committed TOS manifest" in exc.value.message


def test_complete_locks_origin_from_model_file(monkeypatch):
    location = _location(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        release_path="org-10/qwen3/v1",
    )
    content = b"weights"
    files = [_file_entry("weights.bin", content, location.release_path)]
    manifest = _manifest(location, files)
    manifest["origin_huggingface_repo_id"] = "Qwen/Qwen3"
    objects = {
        location.manifest_key: tos_storage.manifest_bytes(manifest),
        files[0]["object_key"]: content,
    }
    monkeypatch.setattr(
        "gpustack.server.tos_bindings.client_from_access",
        lambda access: FakeTos(objects),
    )
    conn = _org_connection(10, 1, "SK-A", monkeypatch)
    job = TosArchiveJob(
        id=9,
        model_file_id=1,
        connection_id=1,
        worker_id=7,
        tos_model="qwen3",
        tos_version="v1",
        state=TosArchiveStateEnum.VERIFYING.value,
        owner_principal_id=10,
        provenance={
            "origin_source": "huggingface",
            "origin_huggingface_repo_id": "Qwen/Qwen3-8B",
        },
    )
    digest = _sha(tos_storage.manifest_bytes(manifest))
    merged = verify_committed_manifest(
        conn,
        job,
        {
            "manifest_sha256": digest,
            "origin_huggingface_repo_id": "attacker/swap",
        },
    )
    assert merged["origin_huggingface_repo_id"] == "Qwen/Qwen3-8B"
    assert merged["manifest_sha256"] == digest


@pytest.mark.asyncio
async def test_server_stamps_gguf_from_manifest_not_client_bool(monkeypatch):
    location = _location(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        model="llama",
        version="q4",
        release_path="org-10/llama/q4",
    )
    gguf = b"GGUF"
    files = [_file_entry("model.gguf", gguf, location.release_path)]
    manifest = _manifest(location, files)
    objects = {
        location.manifest_key: tos_storage.manifest_bytes(manifest),
        files[0]["object_key"]: gguf,
    }
    monkeypatch.setattr(
        "gpustack.server.tos_bindings.client_from_access",
        lambda access: FakeTos(objects),
    )
    conn = _org_connection(10, 4, "SK-A", monkeypatch)
    monkeypatch.setattr(TosConnection, "one_by_id", AsyncMock(return_value=conn))
    model = Model(
        name="gguf-tos",
        source=SourceEnum.TOS,
        tos_connection_id=4,
        tos_endpoint="https://tos-cn-beijing.volces.com",
        tos_region="cn-beijing",
        tos_bucket="weights",
        tos_model="llama",
        tos_version="q4",
        tos_release_path="org-10/llama/q4",
        owner_principal_id=10,
        backend=BackendEnum.VLLM,
        provenance={"gguf": False},
    )
    await stamp_tos_provenance_from_manifest(AsyncMock(), model, owner_principal_id=10)
    assert model.provenance["gguf"] is True
    with pytest.raises(BadRequestException) as exc:
        _reject_tos_gguf_on_transformer_backend(model)
    assert "GGUF" in exc.value.message


@pytest.mark.asyncio
async def test_catalog_fills_manifest_or_marks_unknown(monkeypatch):
    conn = _org_connection(10, 1, "SK-A", monkeypatch)
    location = _location(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        model="qwen3",
        version="v1",
        release_path="org-10/qwen3/v1",
    )
    content = b"weights"
    files = [_file_entry("weights.bin", content, location.release_path)]
    manifest = _manifest(location, files)
    objects = {
        location.manifest_key: tos_storage.manifest_bytes(manifest),
        files[0]["object_key"]: content,
        "org-10/broken/v1/manifest.json": b"not-json",
    }
    fake = FakeTos(objects)
    monkeypatch.setattr(
        tos_archives, "load_visible_connection", AsyncMock(return_value=conn)
    )
    monkeypatch.setattr(tos_archives, "_sdk_client", lambda connection: fake)
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=10,
        org_role=None,
    )
    items = await tos_archives.list_release_catalog(AsyncMock(), ctx, connection_id=1)
    by_version = {(item.tos_model, item.tos_version): item for item in items}
    good = by_version[("qwen3", "v1")]
    assert good.catalog_verified is True
    assert good.gguf is False
    assert good.object_count == 1
    assert good.total_bytes == len(content)
    broken = by_version[("broken", "v1")]
    assert broken.catalog_verified is False
    assert broken.gguf is None
