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

    def list_keys(self, bucket: str, prefix: str):
        marker = prefix or ""
        return [key for key in self.objects if key.startswith(marker)]

    def list_object_infos(self, bucket: str, prefix: str):
        marker = prefix or ""
        infos = []
        for key, content in self.objects.items():
            if not key.startswith(marker):
                continue
            size = len(content) if isinstance(content, (bytes, bytearray)) else 0
            infos.append((key, size))
        return infos

    def object_digest(self, bucket: str, key: str):
        data = self.get_object_bytes(bucket, key)
        return _sha(data), len(data)


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
    with pytest.raises(ValueError, match="tos_model"):
        ModelSource(source=SourceEnum.TOS, tos_bucket="models")
    connected = ModelSource(
        source=SourceEnum.TOS,
        tos_connection_id=1,
        tos_model="qwen3",
        tos_version="v1",
    )
    assert connected.tos_connection_id == 1


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
    dest = tmp_path / "cache"
    paths = downloaders.download_model(
        _tos_source(),
        local_dir=str(dest),
        cache_dir=str(tmp_path / "unused"),
        tos_client=client,
    )
    assert paths == [str(dest)]
    assert (dest / "config.json").read_bytes() == content

    info = downloaders.TosDownloader.get_model_file_info(_tos_source(), client=client)
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
    async def _fake_client(model):
        return object()

    monkeypatch.setattr(policy_utils, "_tos_control_plane_client", _fake_client)
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


def test_upload_single_file_subset(tmp_path):
    location = _location(model="meta-llama-3", version="gguf-q4")
    src = tmp_path / "snap"
    src.mkdir()
    (src / "model.gguf").write_bytes(b"gguf")
    (src / "config.json").write_bytes(b"{}")
    client = FakeTos({})
    published = tos_storage.upload_release(
        location, src, client, relatives=["model.gguf"]
    )
    assert [entry["path"] for entry in published["files"]] == ["model.gguf"]
    assert location.manifest_key in client.objects


def test_upload_resumes_after_objects_without_manifest(tmp_path):
    location = _location()
    src = tmp_path / "snap"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"payload")
    client = FakeTos({})
    scanned = tos_storage.scan_local_release(src, location)
    client.objects[scanned["files"][0]["object_key"]] = b"payload"
    published = tos_storage.upload_release(location, src, client)
    assert location.manifest_key in client.objects
    assert published["files"][0]["sha256"] == scanned["files"][0]["sha256"]


def test_list_releases_stays_inside_prefix():
    location = _location()
    objects = {
        location.manifest_key: b"{}",
        "other-prefix/qwen3/v1/manifest.json": b"{}",
        "lingji-models/qwen3/v2/manifest.json": b"{}",
    }
    listed = tos_storage.list_releases(FakeTos(objects), "models", "lingji-models")
    versions = {(item["tos_model"], item["tos_version"]) for item in listed}
    assert ("qwen3", "v1") in versions
    assert ("qwen3", "v2") in versions
    assert all(item["tos_model"] != "other-prefix/qwen3" for item in listed)


def test_list_catalog_entries_includes_hf_and_weight_dirs():
    # Mirrors the ljyd-algorithm/lingji-models/ layout: manifest releases
    # under releases/, plus HF-style and weight-only repositories, and a
    # media dataset that must stay out of the dropdown.
    sha = "ab" * 32
    objects = {
        "lingji-models/DeepSeek-OCR/config.json": b"{}",
        "lingji-models/DeepSeek-OCR/model.safetensors": b"w",
        "lingji-models/DeepSeek-OCR-2/config.json": b"{}",
        "lingji-models/DeepSeek-OCR-2/model-00001-of-00001.safetensors": b"w",
        (
            "lingji-models/asr/asr-v0.1/"
            "punc_ct-transformer_zh-cn-common-vocab272727-pytorch/config.yaml"
        ): b"x",
        (
            "lingji-models/asr/asr-v0.1/"
            "punc_ct-transformer_zh-cn-common-vocab272727-pytorch/model.pt"
        ): b"w",
        (
            "lingji-models/asr/asr-v0.1/"
            "speech_fsmn_vad_zh-cn-16k-common-pytorch/config.yaml"
        ): b"x",
        (
            "lingji-models/asr/asr-v0.1/"
            "speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt"
        ): b"w",
        (
            "lingji-models/asr/asr-v0.1/"
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/"
            "config.yaml"
        ): b"x",
        (
            "lingji-models/asr/asr-v0.1/"
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/"
            "model.pt"
        ): b"w",
        "lingji-models/bert/bert-20251209/en/config.json": b"{}",
        "lingji-models/bert/bert-20251209/en/pytorch_model.bin": b"w",
        "lingji-models/bert/bert-20251209/zh/config.json": b"{}",
        "lingji-models/bert/bert-20251209/zh/pytorch_model.bin": b"w",
        "lingji-models/bert/roberta-base/config.json": b"{}",
        "lingji-models/bert/roberta-base/pytorch_model.bin": b"w",
        "lingji-models/cv_models/D-FINE_Model/model.pth": b"w",
        "lingji-models/cv_models/ResNest101_Model/weights.bin": b"w",
        "lingji-models/cy_model/best.pt": b"w",
        "lingji-models/cy_model/best_cut_320_k3.onnx": b"w",
        "lingji-models/cy_model/v5n_relu_3cls/args.yaml": b"x",
        "lingji-models/cy_model/v5n_relu_3cls/best_cut_320_3cls.onnx": b"w",
        "lingji-models/cy_model/v5n_relu_3cls/results.csv": b"x",
        "lingji-models/cy_model/v5n_relu_3cls/weights/best.pt": b"w",
        "lingji-models/cy_model/v5n_relu_3cls/weights/last.pt": b"w",
        "lingji-models/cy_model/WFLW/images/0001.jpg": b"i",
        "lingji-models/cy_model/WFLW/images/0002.png": b"i",
        "lingji-models/cy_model/WFLW/annotations.json": b"[]",
        "lingji-models/embeddings/vllm-qwen-embedding-0.6b/config.json": b"{}",
        "lingji-models/embeddings/vllm-qwen-embedding-0.6b/model.safetensors": b"w",
        "lingji-models/llm/llm-qwen3-8b/config.json": b"{}",
        "lingji-models/llm/llm-qwen3-8b/model.safetensors": b"w",
        "lingji-models/react/laya/README.md": b"x",
        "lingji-models/react/laya/configuration.json": b"{}",
        "lingji-models/react/laya/model.safetensors": b"w",
        "lingji-models/react/laya/rl_agent_config.json": b"{}",
        "lingji-models/react/laya/encoder/config.json": b"{}",
        "lingji-models/react/laya/tokenizer/tokenizer.json": b"{}",
        "lingji-models/react/laya/tokenizer/tokenizer_config.json": b"{}",
        "lingji-models/react/laya/multilingual/model.safetensors": b"w",
        "lingji-models/react/laya/multilingual/rl_agent_config.json": b"{}",
        "lingji-models/react/laya/multilingual/encoder/config.json": b"{}",
        "lingji-models/react/laya/multilingual/tokenizer/tokenizer.json": b"{}",
        "lingji-models/react/laya/typed-decisions/model.safetensors": b"w",
        "lingji-models/react/laya/typed-decisions/rl_agent_config.json": b"{}",
        "lingji-models/react/laya/typed-decisions/encoder/config.json": b"{}",
        "lingji-models/react/laya/typed-decisions/tokenizer/tokenizer.json": b"{}",
        "lingji-models/releases/joint-bert-crf/v1/manifest.json": b"{}",
        f"lingji-models/releases/joint-bert-crf/v1/objects/{sha}/weights.bin": b"w",
        "lingji-models/releases/tiny-gpt2/v1/manifest.json": b"{}",
        "lingji-models/releases/vllm-qwen-embedding-0.6b/v1/manifest.json": b"{}",
        "other-prefix/should-not-appear/config.json": b"{}",
        "other-prefix/should-not-appear/model.safetensors": b"w",
    }
    listed = tos_storage.list_catalog_entries(
        FakeTos(objects), "ljyd-algorithm", "lingji-models"
    )
    models = {item["tos_model"] for item in listed}
    expected = {
        "DeepSeek-OCR",
        "DeepSeek-OCR-2",
        "asr/asr-v0.1/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "asr/asr-v0.1/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "asr/asr-v0.1/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "bert/bert-20251209/en",
        "bert/bert-20251209/zh",
        "bert/roberta-base",
        "cv_models/D-FINE_Model",
        "cv_models/ResNest101_Model",
        "cy_model",
        "cy_model/v5n_relu_3cls",
        "embeddings/vllm-qwen-embedding-0.6b",
        "llm/llm-qwen3-8b",
        "react/laya",
        "releases/joint-bert-crf",
        "releases/tiny-gpt2",
        "releases/vllm-qwen-embedding-0.6b",
    }
    assert models == expected
    assert [
        item["tos_model"] for item in listed if item["tos_model"] == "react/laya"
    ] == ["react/laya"]
    laya_internal = {name for name in models if name.startswith("react/laya/")}
    assert laya_internal == set()
    assert "cy_model/v5n_relu_3cls/weights" not in models
    assert "cy_model/WFLW" not in models
    assert "asr" not in models
    assert "asr/asr-v0.1" not in models
    assert "bert" not in models
    assert "bert/bert-20251209" not in models
    assert "releases" not in models
    assert "should-not-appear" not in models
    # Manifest releases keep their real versions; directory models use the sentinel.
    by_model = {item["tos_model"]: item for item in listed}
    assert by_model["releases/tiny-gpt2"]["tos_version"] == "v1"
    assert by_model["releases/tiny-gpt2"]["kind"] == "release"
    assert (
        by_model["DeepSeek-OCR"]["tos_version"] == tos_storage.DIRECTORY_LISTING_VERSION
    )
    assert by_model["DeepSeek-OCR"]["kind"] == "directory"
    # The published releases/ copy must not hide the HF embeddings repo.
    versions = {(item["tos_model"], item["tos_version"]) for item in listed}
    assert ("releases/vllm-qwen-embedding-0.6b", "v1") in versions
    assert (
        "embeddings/vllm-qwen-embedding-0.6b",
        tos_storage.DIRECTORY_LISTING_VERSION,
    ) in versions


def test_list_releases_still_requires_manifest_layout():
    objects = {
        "lingji-models/DeepSeek-OCR/config.json": b"{}",
        "lingji-models/releases/tiny-gpt2/v1/manifest.json": b"{}",
    }
    listed = tos_storage.list_releases(FakeTos(objects), "models", "lingji-models")
    assert listed == [{"tos_model": "releases/tiny-gpt2", "tos_version": "v1"}]


def test_model_id_allows_org_slash_name():
    path = tos_storage.resolve_release_path(
        "meta/llama-3-8b", "v1", prefix="org-a/models"
    )
    assert path == "org-a/models/meta/llama-3-8b/v1"


def test_upload_rejects_corrupt_content_addressed_object(tmp_path):
    location = _location()
    src = tmp_path / "snap"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"payload")
    scanned = tos_storage.scan_local_release(src, location)
    client = FakeTos({scanned["files"][0]["object_key"]: b"tampered-bytes"})
    tos_storage.upload_release(location, src, client)
    assert client.objects[scanned["files"][0]["object_key"]] == b"payload"


def test_https_endpoint_rejects_private_and_localhost():
    with pytest.raises(tos_storage.TosStorageError, match="private"):
        tos_storage._https_endpoint("https://10.1.2.3")
    with pytest.raises(tos_storage.TosStorageError, match="not allowed"):
        tos_storage._https_endpoint("https://localhost")
    with pytest.raises(tos_storage.TosStorageError, match="not allowed"):
        tos_storage._https_endpoint("https://metadata.google.internal")


def test_upload_rejects_path_escape(tmp_path):
    location = _location()
    src = tmp_path / "snap"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"x")
    with pytest.raises(tos_storage.TosStorageError):
        tos_storage.upload_release(
            location, src, FakeTos({}), relatives=["../secret.bin"]
        )


def test_resolve_directory_listing_omits_sentinel_version():
    assert (
        tos_storage.resolve_release_path("react/laya", "dir", prefix="lingji-models")
        == "lingji-models/react/laya"
    )
    assert (
        tos_storage.resolve_release_path(
            "releases/tiny-gpt2", "v1", prefix="lingji-models"
        )
        == "lingji-models/releases/tiny-gpt2/v1"
    )
    location = _location(
        model="react/laya",
        version="dir",
        release_path=tos_storage.resolve_release_path(
            "react/laya", "dir", prefix="lingji-models"
        ),
    )
    assert tos_storage.is_directory_package(location)
    assert location.uri == "tos://models/lingji-models/react/laya"


def test_download_directory_pulls_entire_root(tmp_path):
    location = _location(
        model="react/laya",
        version="dir",
        release_path="lingji-models/react/laya",
    )
    objects = {
        "lingji-models/react/laya/configuration.json": b"cfg",
        "lingji-models/react/laya/model.safetensors": b"weights",
        "lingji-models/react/laya/encoder/config.json": b"enc",
        "lingji-models/react/laya/": b"",
        "lingji-models/react/laya/.tos-tmp/partial": b"tmp",
        "lingji-models/react/laya-other/model.bin": b"not-in-package",
    }
    dest = tmp_path / "laya"
    client = FakeTos(objects)
    paths = tos_storage.download_package(location, dest, client)
    assert paths == [str(dest.absolute())]
    assert (dest / "configuration.json").read_bytes() == b"cfg"
    assert (dest / "model.safetensors").read_bytes() == b"weights"
    assert (dest / "encoder" / "config.json").read_bytes() == b"enc"
    assert not (dest / "model.bin").exists()
    assert not (dest / ".tos-tmp" / "partial").exists()
    listed = tos_storage.list_directory_package_files(location, client)
    assert [entry["path"] for entry in listed] == [
        "configuration.json",
        "encoder/config.json",
        "model.safetensors",
    ]
    # Second pull is a no-op once the inventory marker matches.
    again = tos_storage.download_package(location, dest, client)
    assert again == paths


def test_tos_downloader_directory_uses_root_not_manifest(tmp_path):
    model = _tos_source(
        tos_model="react/laya",
        tos_version="dir",
        tos_release_path="lingji-models/react/laya",
    )
    objects = {
        "lingji-models/react/laya/configuration.json": b"cfg",
        "lingji-models/react/laya/weights/model.safetensors": b"weights",
    }
    client = FakeTos(objects)
    entries = downloaders.TosDownloader.get_model_file_info(model, client=client)
    assert {entry.rfilename for entry in entries} == {
        "configuration.json",
        "weights/model.safetensors",
    }
    dest = tmp_path / "pkg"
    paths = downloaders.TosDownloader.download(model, local_dir=dest, client=client)
    assert paths == [str(dest.absolute())]
    assert (dest / "configuration.json").read_bytes() == b"cfg"
    assert (dest / "weights" / "model.safetensors").read_bytes() == b"weights"
    assert not (dest / "manifest.json").exists()


def test_validate_manifest_accepts_releases_prefix_catalog_id():
    location = _location(
        model="releases/tiny-gpt2",
        version="v1",
        release_path="lingji-models/releases/tiny-gpt2/v1",
    )
    content = b"weights"
    files = [_file_entry("weights.bin", content, location.release_path)]
    manifest = _manifest(location, files)
    manifest["model"] = "tiny-gpt2"
    validated = tos_storage.validate_manifest(manifest, location)
    assert validated["model"] == "tiny-gpt2"
    client = FakeTos(
        {
            location.manifest_key: tos_storage.manifest_bytes(manifest),
            files[0]["object_key"]: content,
        }
    )
    fetched = tos_storage.fetch_manifest(location, client)
    assert fetched["version"] == "v1"
    assert [entry["path"] for entry in fetched["files"]] == ["weights.bin"]
