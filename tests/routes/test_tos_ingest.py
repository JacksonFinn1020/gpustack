import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.exceptions import InvalidException, register_handlers
from gpustack.routes.tos import (
    TosIngestRequest,
    ingest_release,
    model_sources,
    router,
)
from gpustack.schemas.models import ModelSource, SourceEnum
from gpustack.worker import tos_storage
from tests.worker.test_tos_storage import FakeTos


def test_source_dropdown_includes_tos_fields():
    sources = {item["source"]: item for item in model_sources()}
    assert set(sources) == {"huggingface", "model_scope", "local_path", "tos"}
    names = [field["name"] for field in sources["tos"]["fields"]]
    assert names == [
        "tos_endpoint",
        "tos_region",
        "tos_bucket",
        "tos_model",
        "tos_version",
        "tos_release_path",
    ]
    assert sources["tos"]["deploy_path"] == "/tos-deploy"


def test_ingest_api_returns_model_create_payload(tmp_path):
    src = tmp_path / "hub-snapshot"
    src.mkdir()
    (src / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}')
    req = TosIngestRequest(
        local_path=str(src),
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
        tos_release_path="lingji-models/qwen3/v1",
    )
    client = FakeTos({})
    result = ingest_release(req, client=client)
    assert result.source.value == "tos"
    assert result.model_create["source"] == "tos"
    assert result.model_create["tos_release_path"] == "lingji-models/qwen3/v1"
    assert not any("secret" in key or "access" in key for key in result.model_create)
    assert result.files == ["config.json"]
    assert result.uri == "tos://models/lingji-models/qwen3/v1"
    dest = tmp_path / "pulled"
    location = tos_storage.location_from_source(
        ModelSource(
            source=SourceEnum.TOS,
            tos_endpoint=req.tos_endpoint,
            tos_region=req.tos_region,
            tos_bucket=req.tos_bucket,
            tos_model=req.tos_model,
            tos_version=req.tos_version,
            tos_release_path=req.tos_release_path,
        )
    )
    paths = tos_storage.download_release(location, dest, client)
    assert paths == [str(dest)]
    assert (dest / "config.json").read_text() == (src / "config.json").read_text()


def test_ingest_rejects_changed_bytes_for_same_version(tmp_path):
    first = tmp_path / "one"
    first.mkdir()
    (first / "weights.bin").write_bytes(b"one")
    req = TosIngestRequest(
        local_path=str(first),
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
    )
    client = FakeTos({})
    ingest_release(req, client=client)
    second = tmp_path / "two"
    second.mkdir()
    (second / "weights.bin").write_bytes(b"two")
    req.local_path = str(second)
    with pytest.raises(InvalidException) as exc_info:
        ingest_release(req, client=client)
    assert "already published" in exc_info.value.message


def test_ingest_rejects_missing_directory(tmp_path):
    req = TosIngestRequest(
        local_path=str(tmp_path / "missing"),
        tos_endpoint="https://tos-cn-shanghai.volces.com",
        tos_region="cn-shanghai",
        tos_bucket="models",
        tos_model="qwen3",
        tos_version="v1",
    )
    with pytest.raises(InvalidException):
        ingest_release(req, client=FakeTos({}))


def test_tos_deploy_page_lists_source_dropdown():
    app = FastAPI()
    register_handlers(app)
    app.include_router(router, prefix="/v2/tos")
    from gpustack.routes.tos import ui_router

    app.include_router(ui_router)
    client = TestClient(app)
    page = client.get("/tos-deploy")
    assert page.status_code == 200
    assert "TOS model source" in page.text
    assert 'id="source"' in page.text
    assert 'value="tos"' in page.text
    assert 'data-tos="tos_endpoint"' in page.text
    assert 'data-tos="tos_version"' in page.text
    assert "gpu_ids" in page.text
    assert "gpu_selector" in page.text
    assert "modelCreateFromForm" in page.text
    assert 'API + "/models"' in page.text
    assert "Hugging Face, ModelScope, and local_path stay on the main Deploy form" in (
        page.text
    )
    listed = client.get("/v2/tos/sources")
    assert listed.status_code == 200
    assert any(item["source"] == "tos" for item in listed.json()["items"])
