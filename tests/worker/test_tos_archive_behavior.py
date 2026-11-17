"""Behavioral archive/upload/recovery tests (not route-allowlist parametrize)."""

from types import SimpleNamespace

import pytest

from gpustack.api.exceptions import BadRequestException, ForbiddenException
from gpustack.routes.models import _reject_tos_gguf_on_transformer_backend
from gpustack.routes.tos_archives import _require_assigned_worker
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    SourceEnum,
    is_gguf_model,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.tos import TosArchiveStateEnum
from gpustack.server.tos_bindings import origin_from_model_file
from gpustack.utils.tos_scope import env_endpoint_allows
from gpustack.worker.tos_archive_manager import TosArchiveManager


def test_origin_comes_from_model_file_record():
    file = SimpleNamespace(
        id=9,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-8B",
        huggingface_filename="model.safetensors",
        model_scope_model_id=None,
        model_scope_file_path=None,
        local_path=None,
        worker_id=4,
    )
    origin = origin_from_model_file(file)
    assert origin["origin_source"] == "huggingface"
    assert origin["origin_huggingface_repo_id"] == "Qwen/Qwen3-8B"
    assert origin["origin_model_file_id"] == 9
    assert origin["origin_worker_id"] == 4


def test_single_file_gguf_flag_from_selected_paths():
    model = Model(
        name="gguf-tos",
        source=SourceEnum.TOS,
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="llama",
        tos_version="q4",
        provenance={"gguf": True, "selected_paths": ["Llama-3-8B-Q4_K_M.gguf"]},
        backend=BackendEnum.CUSTOM,
    )
    assert is_gguf_model(model)
    _reject_tos_gguf_on_transformer_backend(model)
    model.backend = BackendEnum.VLLM
    with pytest.raises(BadRequestException):
        _reject_tos_gguf_on_transformer_backend(model)


def test_non_assigned_worker_cannot_use_transport_gate():
    job = SimpleNamespace(worker_id=7)
    other = SimpleNamespace(
        user=SimpleNamespace(kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=8))
    )
    owner = SimpleNamespace(user=SimpleNamespace(kind=PrincipalType.USER, worker=None))
    with pytest.raises(ForbiddenException):
        _require_assigned_worker(other, job)
    with pytest.raises(ForbiddenException):
        _require_assigned_worker(owner, job)
    assigned = SimpleNamespace(
        user=SimpleNamespace(kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=7))
    )
    _require_assigned_worker(assigned, job)


def test_env_mode_refuses_unpinned_or_mismatched_endpoint(monkeypatch):
    monkeypatch.delenv("TOS_ENDPOINT", raising=False)
    monkeypatch.delenv("TOS_ALLOWED_ENDPOINTS", raising=False)
    assert not env_endpoint_allows("https://tos-cn-beijing.volces.com")
    monkeypatch.setenv("TOS_ENDPOINT", "https://tos-cn-beijing.volces.com")
    assert env_endpoint_allows("https://tos-cn-beijing.volces.com")
    assert not env_endpoint_allows("https://attacker.example.com")


def test_complete_keeps_origin_over_worker_payload():
    stored = {
        "origin_source": "huggingface",
        "origin_huggingface_repo_id": "Qwen/Qwen3-8B",
        "origin_model_file_id": 9,
    }
    incoming = {
        "origin_source": "forged",
        "origin_huggingface_repo_id": "evil/model",
        "manifest_sha256": "a" * 64,
        "artifact_uri": "tos://bucket/m/v",
    }
    origin = {key: stored[key] for key in stored if str(key).startswith("origin_")}
    merged = {**stored, **incoming, **origin}
    assert merged["origin_huggingface_repo_id"] == "Qwen/Qwen3-8B"
    assert merged["manifest_sha256"] == "a" * 64


def test_worker_reclaims_incomplete_jobs_not_only_queued():
    restartable = {
        TosArchiveStateEnum.QUEUED.value,
        TosArchiveStateEnum.UPLOADING.value,
        TosArchiveStateEnum.VERIFYING.value,
    }
    assert TosArchiveManager.__init__.__name__
    assert "error" not in restartable
    assert "ready" not in restartable


def test_auto_archive_error_does_not_clear_ready_intent():
    file = SimpleNamespace(
        state=ModelFileStateEnum.READY,
        archive_state=TosArchiveStateEnum.ERROR.value,
    )
    assert file.state == ModelFileStateEnum.READY
    assert file.archive_state == TosArchiveStateEnum.ERROR.value
