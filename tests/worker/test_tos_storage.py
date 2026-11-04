import hashlib
import json
from pathlib import Path

import pytest

from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    Model,
    ModelSource,
    SourceEnum,
)
from gpustack.policies import utils as policy_utils
from gpustack.scheduler.evaluator import evaluate_model_metadata
from gpustack.scheduler.scheduler import evaluate_pretrained_config
from gpustack.utils import hub
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
        self.objects = dict(objects)

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise tos_storage.TosStorageError("missing object") from exc

    def try_get_object_bytes(self, bucket: str, key: str):
        return self.objects.get(key)

    def object_exists(self, bucket: str, key: str) -> bool:
        return key in self.objects

    def upload_file(self, bucket: str, key: str, path: Path) -> None:
        self.objects[key] = Path(path).read_bytes()

    def put_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        forbid_overwrite: bool = False,
    ) -> None:
        if forbid_overwrite and key in self.objects:
            raise tos_storage.TosStorageError("object exists")
        self.objects[key] = content

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


def _tos_model() -> Model:
    return Model(
        name="qwen3-tos",
        source=SourceEnum.TOS,
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
        backend=BackendEnum.VLLM,
    )


@pytest.mark.asyncio
async def test_tos_evaluate_pretrained_config_skips_server_architecture():
    model = _tos_model()
    updated = await evaluate_pretrained_config(model)
    assert updated is True
    assert model.categories == [CategoryEnum.LLM]


@pytest.mark.asyncio
async def test_tos_evaluate_model_metadata_is_compatible(config):
    compatible, messages = await evaluate_model_metadata(config, _tos_model(), [])
    assert compatible is True
    assert messages == []


def test_match_file_uses_tos_cache_dir(tmp_path, monkeypatch):
    source = _tos_source()
    dest = tos_storage.cache_dir_for(source, str(tmp_path / "cache"))
    files = [
        hub.FileEntry("weights.bin", 11, sha256="a" * 64),
    ]
    size, paths = hub.match_file_and_calculate_size(
        files, source, cache_dir=str(tmp_path / "cache")
    )
    assert size == 11
    assert paths == [str(dest)]

    monkeypatch.setenv("TOS_MODEL_ROOT", str(tmp_path / "data" / "models"))
    size, paths = hub.match_file_and_calculate_size(
        files, source, cache_dir=str(tmp_path / "cache")
    )
    assert paths == [str(tmp_path / "data" / "models" / "qwen3" / "v1")]


def test_upload_then_download_round_trip(tmp_path):
    location = _location()
    src = tmp_path / "snapshot"
    src.mkdir()
    (src / "config.json").write_bytes(b'{"architectures":["Qwen3ForCausalLM"]}')
    (src / "weights.bin").write_bytes(b"fake-weights")
    client = FakeTos({})
    published = tos_storage.upload_release(location, src, client)
    assert published["model"] == "qwen3"
    assert {entry["path"] for entry in published["files"]} == {
        "config.json",
        "weights.bin",
    }
    dest = tmp_path / "pulled"
    paths = tos_storage.download_release(location, dest, client)
    assert paths == [str(dest)]
    assert (dest / "weights.bin").read_bytes() == b"fake-weights"


def test_upload_rejects_changed_immutable_version(tmp_path):
    location = _location()
    first = tmp_path / "a"
    first.mkdir()
    (first / "weights.bin").write_bytes(b"one")
    client = FakeTos({})
    tos_storage.upload_release(location, first, client)
    second = tmp_path / "b"
    second.mkdir()
    (second / "weights.bin").write_bytes(b"two")
    with pytest.raises(tos_storage.TosStorageError, match="already published"):
        tos_storage.upload_release(location, second, client)


def test_tos_pretrained_config_reads_local_cache(tmp_path, config):
    dest = tos_storage.cache_dir_for(_tos_source(), config.cache_dir)
    dest.mkdir(parents=True)
    (dest / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":16}'
    )
    model = _tos_model()
    model.tos_release_path = "lingji-models/qwen3/v1"
    parsed = hub.get_pretrained_config(model)
    assert parsed.architectures == ["Qwen3ForCausalLM"]
    assert parsed.hidden_size == 16
    assert hub.read_repo_file_content(model, "config.json")["model_type"] == "qwen3"


@pytest.mark.asyncio
async def test_tos_vram_uses_manifest_size(monkeypatch):
    monkeypatch.setattr(
        downloaders.TosDownloader,
        "get_model_file_info",
        classmethod(
            lambda cls, model, client=None: [
                hub.FileEntry("weights.bin", 5 * 1024**3),
                hub.FileEntry("config.json", 128),
            ]
        ),
    )
    size = await policy_utils._weight_size_without_cache(
        _tos_model(), workers=None, is_diffusion=False
    )
    assert size == 5 * 1024**3 + 128


def test_scan_skips_release_metadata(tmp_path):
    location = _location()
    src = tmp_path / "tree"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"w")
    (src / tos_storage.LOCAL_COMPLETE_NAME).write_text("{}")
    manifest = tos_storage.scan_local_release(src, location)
    assert [entry["path"] for entry in manifest["files"]] == ["weights.bin"]


def test_upload_is_idempotent_for_same_bytes(tmp_path):
    location = _location()
    src = tmp_path / "snap"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"same")
    client = FakeTos({})
    first = tos_storage.upload_release(location, src, client)
    again = tos_storage.upload_release(location, src, client)
    assert first == again
