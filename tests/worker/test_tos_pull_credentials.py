"""Inline TOS pull must not borrow worker env; env mode must pin the endpoint."""

import json

import pytest

from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.models import ModelSource, SourceEnum
from gpustack.worker import downloaders, tos_storage
from tests.worker.test_tos_storage import FakeTos, _file_entry, _location, _manifest


def _inline_source():
    return ModelFile(
        source=SourceEnum.TOS,
        tos_connection_id=3,
        tos_endpoint="https://tos-cn-beijing.volces.com",
        tos_region="cn-beijing",
        tos_bucket="weights",
        tos_model="qwen3",
        tos_version="v1",
        tos_release_path="org-a/qwen3/v1",
        provenance=None,
    )


def test_inline_connection_pull_works_without_worker_tos_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TOS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TOS_SECRET_KEY", raising=False)
    monkeypatch.delenv("TOS_ENDPOINT", raising=False)
    location = _location(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="weights",
        release_path="org-a/qwen3/v1",
    )
    content = b"ok"
    files = [_file_entry("config.json", content, location.release_path)]
    manifest = _manifest(location, files)
    manifest["origin_huggingface_repo_id"] = "Qwen/Qwen3-8B"
    manifest["origin_source"] = "huggingface"
    objects = {
        location.manifest_key: json.dumps(manifest).encode(),
        files[0]["object_key"]: content,
    }
    client = FakeTos(objects)
    dest = tmp_path / "cache"
    source = _inline_source()
    with pytest.raises(tos_storage.TosStorageError, match="pull transport"):
        downloaders.TosDownloader.download(source, local_dir=str(dest))
    paths = downloaders.TosDownloader.download(
        source, local_dir=str(dest), client=client
    )
    assert paths == [str(dest)]
    assert source.provenance["origin_huggingface_repo_id"] == "Qwen/Qwen3-8B"
    assert source.provenance["gguf"] is False


def test_env_mode_rejects_unpinned_arbitrary_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TOS_ACCESS_KEY", "AK-ENV")
    monkeypatch.setenv("TOS_SECRET_KEY", "SK-ENV")
    monkeypatch.delenv("TOS_ENDPOINT", raising=False)
    monkeypatch.delenv("TOS_ALLOWED_ENDPOINTS", raising=False)
    source = ModelSource(
        source=SourceEnum.TOS,
        tos_endpoint="https://attacker.example",
        tos_region="cn-beijing",
        tos_bucket="weights",
        tos_model="qwen3",
        tos_version="v1",
        tos_release_path="qwen3/v1",
    )
    with pytest.raises(tos_storage.TosStorageError, match="pinned"):
        downloaders.TosDownloader.download(source, local_dir=str(tmp_path / "out"))


def test_sdk_client_does_not_silently_use_env_for_inline(monkeypatch):
    monkeypatch.setenv("TOS_ACCESS_KEY", "AK-ENV")
    monkeypatch.setenv("TOS_SECRET_KEY", "SK-ENV")
    with pytest.raises(tos_storage.TosStorageError, match="missing"):
        tos_storage.SdkTosClient(
            "https://tos-cn-beijing.volces.com",
            "cn-beijing",
            use_env=False,
        )
    with pytest.raises(tos_storage.TosStorageError, match="TOS_ENDPOINT"):
        tos_storage.SdkTosClient(
            "https://tos-cn-beijing.volces.com",
            "cn-beijing",
            use_env=True,
        )
