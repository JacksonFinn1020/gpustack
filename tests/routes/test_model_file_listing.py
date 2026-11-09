from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastapi import HTTPException

from gpustack.api.exceptions import (
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_files as routes
from gpustack.schemas.filesystem import ArchiveListingFile, ArchiveListingResponse
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.utils.model_file_listing import filter_worker_listing_items


def _ctx(principal_id=10, role=OrgRole.OWNER):
    return TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=role,
    )


def _ready_file(**kwargs):
    fields = dict(
        id=5,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3",
        state=ModelFileStateEnum.READY,
        worker_id=7,
        owner_principal_id=10,
        cluster_id=3,
        resolved_paths=["/cache/qwen"],
        deleted_at=None,
    )
    fields.update(kwargs)
    return ModelFile(**fields)


@pytest.mark.asyncio
async def test_other_org_cannot_list_files(monkeypatch):
    monkeypatch.setattr(
        ModelFile,
        "one_by_id",
        AsyncMock(return_value=_ready_file(owner_principal_id=99)),
    )
    with pytest.raises(NotFoundException):
        await routes.list_model_file_files(AsyncMock(), _ctx(10), 5)


@pytest.mark.asyncio
async def test_non_ready_file_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ModelFile,
        "one_by_id",
        AsyncMock(return_value=_ready_file(state=ModelFileStateEnum.DOWNLOADING)),
    )
    with pytest.raises(InvalidException) as exc:
        await routes.list_model_file_files(AsyncMock(), _ctx(), 5)
    assert exc.value.details["error_code"] == "not_ready"


@pytest.mark.asyncio
async def test_unreachable_worker_is_not_empty_success(monkeypatch):
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=_ready_file()))
    worker = SimpleNamespace(id=7, deleted_at=None, state=WorkerStateEnum.UNREACHABLE)
    monkeypatch.setattr(routes.Worker, "one_by_id", AsyncMock(return_value=worker))
    with pytest.raises(ServiceUnavailableException) as exc:
        await routes.list_model_file_files(AsyncMock(), _ctx(), 5)
    assert exc.value.details["error_code"] == "worker_unreachable"


@pytest.mark.asyncio
async def test_missing_cache_is_not_empty_success(monkeypatch):
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=_ready_file()))
    worker = SimpleNamespace(id=7, deleted_at=None, state=WorkerStateEnum.READY)
    monkeypatch.setattr(routes.Worker, "one_by_id", AsyncMock(return_value=worker))

    class _Boom:
        async def __aenter__(self):
            raise RuntimeError("Worker request failed: 7 GET x status=404, error=gone")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient",
        lambda: _Boom(),
    )
    with pytest.raises(InvalidException) as exc:
        await routes.list_model_file_files(AsyncMock(), _ctx(), 5)
    assert exc.value.details["error_code"] == "cache_missing"


@pytest.mark.asyncio
async def test_untrusted_worker_payload_rejected(monkeypatch):
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=_ready_file()))
    worker = SimpleNamespace(id=7, deleted_at=None, state=WorkerStateEnum.READY)
    monkeypatch.setattr(routes.Worker, "one_by_id", AsyncMock(return_value=worker))

    payload = ArchiveListingResponse(
        items=[
            ArchiveListingFile(path="/etc/passwd", size=1),
            ArchiveListingFile(path="../x", size=1),
        ]
    )

    class _Client:
        async def __aenter__(self):
            mock = MagicMock()
            mock.list_archive_files = AsyncMock(return_value=payload)
            return mock

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient",
        lambda: _Client(),
    )
    with pytest.raises(InvalidException) as exc:
        await routes.list_model_file_files(AsyncMock(), _ctx(), 5)
    assert exc.value.details["error_code"] == "path_escape"


@pytest.mark.asyncio
async def test_ready_directory_returns_relative_gguf(monkeypatch):
    monkeypatch.setattr(ModelFile, "one_by_id", AsyncMock(return_value=_ready_file()))
    worker = SimpleNamespace(id=7, deleted_at=None, state=WorkerStateEnum.READY)
    monkeypatch.setattr(routes.Worker, "one_by_id", AsyncMock(return_value=worker))
    payload = ArchiveListingResponse(
        items=[
            ArchiveListingFile(path="config.json", size=2),
            ArchiveListingFile(path="weights/Llama-3-8B-Q4_K_M.gguf", size=4),
        ],
        truncated=False,
        next_after=None,
        root_kind="directory",
    )

    class _Client:
        async def __aenter__(self):
            mock = MagicMock()
            mock.list_archive_files = AsyncMock(return_value=payload)
            return mock

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient",
        lambda: _Client(),
    )
    result = await routes.list_model_file_files(AsyncMock(), _ctx(), 5)
    assert result.root_kind == "directory"
    assert [item.path for item in result.items] == [
        "config.json",
        "weights/Llama-3-8B-Q4_K_M.gguf",
    ]


def _worker_request(tmp_path, model_file, worker_id=7, cache_dir=None):
    cache = Path(cache_dir) if cache_dir else tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    files = MagicMock()
    files.get.return_value = model_file
    req = MagicMock()
    req.app.state.config = SimpleNamespace(cache_dir=str(cache))
    req.app.state.serve_manager = SimpleNamespace(
        _clientset=SimpleNamespace(model_files=files),
        _worker_id=worker_id,
    )
    return req, cache


@pytest.mark.asyncio
async def test_worker_listing_missing_dir_is_404(tmp_path):
    from gpustack.routes.worker import filesystem as worker_fs

    model_file = _ready_file(resolved_paths=[str(tmp_path / "cache" / "gone")])
    req, _ = _worker_request(tmp_path, model_file)
    with pytest.raises(HTTPException) as exc:
        await worker_fs.archive_listing(req, model_file_id=5, after=None, limit=10)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_worker_listing_rejects_system_directory(tmp_path):
    from gpustack.routes.worker import filesystem as worker_fs

    outside = tmp_path / "tmp"
    outside.mkdir()
    (outside / "secret").write_bytes(b"x")
    model_file = _ready_file(resolved_paths=[str(outside)])
    req, _ = _worker_request(tmp_path, model_file)
    with pytest.raises(HTTPException) as exc:
        await worker_fs.archive_listing(req, model_file_id=5, after=None, limit=10)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_worker_listing_rejects_other_workers_file(tmp_path):
    from gpustack.routes.worker import filesystem as worker_fs

    model_file = _ready_file(worker_id=99)
    req, _ = _worker_request(tmp_path, model_file, worker_id=7)
    with pytest.raises(HTTPException) as exc:
        await worker_fs.archive_listing(req, model_file_id=5, after=None, limit=10)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_worker_listing_rejects_non_ready(tmp_path):
    from gpustack.routes.worker import filesystem as worker_fs

    model_file = _ready_file(state=ModelFileStateEnum.DOWNLOADING)
    req, _ = _worker_request(tmp_path, model_file)
    with pytest.raises(HTTPException) as exc:
        await worker_fs.archive_listing(req, model_file_id=5, after=None, limit=10)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_worker_listing_hf_ready_under_cache(tmp_path):
    from gpustack.routes.worker import filesystem as worker_fs

    req, cache = _worker_request(tmp_path, _ready_file())
    root = cache / "huggingface" / "Qwen" / "Qwen3"
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "Q4.gguf").write_bytes(b"GGUF")
    model_file = _ready_file(resolved_paths=[str(root)])
    req.app.state.serve_manager._clientset.model_files.get.return_value = model_file
    result = await worker_fs.archive_listing(req, model_file_id=5, after=None, limit=10)
    assert result.root_kind == "directory"
    assert "weights/Q4.gguf" in [item.path for item in result.items]


def test_filter_keeps_only_archive_relatives():
    kept = filter_worker_listing_items(
        [{"path": "a.gguf", "size": 2}, {"path": "/abs", "size": 2}]
    )
    assert [item.path for item in kept] == ["a.gguf"]
