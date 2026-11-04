import hashlib
import json
from pathlib import Path

import pytest

from gpustack.schemas.models import ModelSource, SourceEnum
from gpustack.worker import downloaders, tos_storage


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _location(**overrides) -> tos_storage.TosLocation:
    fields = dict(
        endpoint="https://tos-cn-shanghai.volces.com",
        region="cn-shanghai",
        bucket="models",
        model="qwen3",
        version="v1",
        release_path="lingji-models/qwen3/v1",
    )
    fields.update(overrides)
    return tos_storage.TosLocation(**fields)


def _file_entry(path: str, content: bytes, release_path: str) -> dict:
    digest = _sha(content)
    return {
        "path": path,
        "size": len(content),
        "sha256": digest,
        "object_key": tos_storage.object_key_for(release_path, digest, path),
        "content": content,
    }


def _manifest(location, files):
    return {
        "schema_version": 1,
        "model": location.model,
        "version": location.version,
        "files": [
            {k: v for k, v in entry.items() if k != "content"} for entry in files
        ],
    }


class FakeTos:
    def __init__(self, objects):
        self.objects = objects

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise tos_storage.TosStorageError("missing object") from exc

    def download_object(self, bucket: str, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get_object_bytes(bucket, key))


def _tos_source(**overrides) -> ModelSource:
    fields = dict(
        source=SourceEnum.TOS,
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
        tos_release_path="lingji-models/qwen3/v1",
    )
    fields.update(overrides)
    return ModelSource(**fields)


def test_release_path_and_object_key():
    assert (
        tos_storage.resolve_release_path("qwen3", "v1", prefix="lingji-models")
        == "lingji-models/qwen3/v1"
    )
    assert (
        tos_storage.resolve_release_path(
            "qwen3", "v1", release_path="releases/{model}/{version}"
        )
        == "releases/qwen3/v1"
    )
    digest = "a" * 64
    assert (
        tos_storage.object_key_for("lingji-models/qwen3/v1", digest, "config.json")
        == f"lingji-models/qwen3/v1/objects/{digest}/config.json"
    )


def test_validate_manifest_rejects_wrong_object_key():
    location = _location()
    content = b'{"a":1}'
    files = [_file_entry("config.json", content, location.release_path)]
    files[0]["object_key"] = "other/objects/" + files[0]["sha256"] + "/config.json"
    with pytest.raises(tos_storage.TosStorageError, match="object path"):
        tos_storage.validate_manifest(_manifest(location, files), location)


def test_download_verifies_sha_and_writes_local_tree(tmp_path):
    location = _location()
    weights = b"fake-weights"
    files = [_file_entry("weights.bin", weights, location.release_path)]
    manifest = _manifest(location, files)
    objects = {
        location.manifest_key: json.dumps(manifest).encode(),
        files[0]["object_key"]: weights,
    }
    dest = tmp_path / "models" / "qwen3" / "v1"
    progress = []
    paths = tos_storage.download_release(
        location,
        dest,
        FakeTos(objects),
        progress_callback=lambda name, n: progress.append((name, n)),
    )
    assert paths == [str(dest)]
    assert (dest / "weights.bin").read_bytes() == weights
    assert progress == [("weights.bin", len(weights))]
    # Completed tree is reused without rewriting.
    again = tos_storage.download_release(location, dest, FakeTos(objects))
    assert again == [str(dest)]


def test_download_rejects_bad_sha(tmp_path):
    location = _location()
    files = [_file_entry("weights.bin", b"expected", location.release_path)]
    manifest = _manifest(location, files)
    objects = {
        location.manifest_key: json.dumps(manifest).encode(),
        files[0]["object_key"]: b"tampered",
    }
    with pytest.raises(tos_storage.TosStorageError, match="SHA256"):
        tos_storage.download_release(location, tmp_path / "out", FakeTos(objects))


def test_tos_source_validation():
    source = _tos_source()
    assert source.readable_source == "tos://models/lingji-models/qwen3/v1"
    assert source.model_source_index
    with pytest.raises(ValueError, match="tos_endpoint"):
        ModelSource(source=SourceEnum.TOS, tos_bucket="models")


def test_other_sources_still_require_their_fields():
    with pytest.raises(ValueError, match="huggingface_repo_id"):
        ModelSource(source=SourceEnum.HUGGING_FACE)
    with pytest.raises(ValueError, match="model_scope_model_id"):
        ModelSource(source=SourceEnum.MODEL_SCOPE)
    with pytest.raises(ValueError, match="local_path"):
        ModelSource(source=SourceEnum.LOCAL_PATH)


def test_download_model_tos_uses_worker_client(tmp_path, monkeypatch):
    location = _location()
    content = b"ok"
    files = [_file_entry("config.json", content, location.release_path)]
    manifest = _manifest(location, files)
    objects = {
        location.manifest_key: json.dumps(manifest).encode(),
        files[0]["object_key"]: content,
    }
    client = FakeTos(objects)
    monkeypatch.setattr(
        tos_storage,
        "SdkTosClient",
        lambda endpoint, region: client,
    )
    dest = tmp_path / "cache"
    paths = downloaders.download_model(
        _tos_source(),
        local_dir=str(dest),
        cache_dir=str(tmp_path / "unused"),
    )
    assert paths == [str(dest)]
    assert (dest / "config.json").read_bytes() == content

    info = downloaders.get_model_file_info(_tos_source())
    assert [entry.rfilename for entry in info] == ["config.json"]
    assert info[0].size == len(content)


def test_unsupported_source_still_rejected():
    class Other:
        source = "s3"

    with pytest.raises(ValueError, match="Unsupported model source"):
        downloaders.get_model_file_info(Other())
