import json
from pathlib import Path

import pytest
import yaml

from bridge import (
    ConvertError,
    UnsupportedRuntimeError,
    apply_body,
    convert_release,
    format_plan,
    load_json,
    required_release_fields,
)
from convert import main as convert_main
from plan import main as plan_main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "model_deploy"


def _inputs(name: str):
    root = FIXTURES / name
    deploy = load_json(root / "deploy.json")
    catalog = load_json(root / "model-catalog.json")
    service = (
        load_json(root / "service.json") if (root / "service.json").exists() else None
    )
    return deploy, catalog, service


def test_convert_vllm_standard_local_path_and_offline_env():
    deploy, catalog, service = _inputs("vllm-standard")
    release = convert_release(
        deploy,
        catalog,
        service=service,
        model="vllm-standard",
        version="v1",
    )
    assert required_release_fields(release) == []
    create = release["model_create"]
    assert create["source"] == "local_path"
    assert create["local_path"] == "/data/models/vllm-standard/v1"
    assert create["backend"] == "vLLM"
    assert create["name"] == "vllm-standard"
    assert create["env"]["HF_HUB_OFFLINE"] == "1"
    assert create["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "gpu_selector" not in create
    assert release["probe"]["health_path"] == "/v1/models"
    assert apply_body(release)["source"] == "local_path"


def test_convert_sglang_tp_extra_args_and_gpu_selector():
    deploy, catalog, service = _inputs("sglang-standard")
    release = convert_release(
        deploy,
        catalog,
        service=service,
        model="sglang-standard",
        version="v2",
        worker="gpu-worker-a",
    )
    create = release["model_create"]
    assert create["backend"] == "SGLang"
    assert create["local_path"] == "/data/models/sglang-standard/v2"
    assert "--tp=2" in create["backend_parameters"]
    assert "--context-length=8192" in create["backend_parameters"]
    assert create["gpu_selector"]["gpu_ids"] == [
        "gpu-worker-a:cuda:0",
        "gpu-worker-a:cuda:1",
    ]
    assert create["env"]["EXTRA_FLAG"] == "1"
    assert apply_body(release)["backend"] == "SGLang"


@pytest.mark.parametrize(
    "fixture,model",
    [
        ("custom-http-standard", "custom-http-standard"),
        ("python-adapter-standard", "laya-standard"),
    ],
)
def test_unsupported_runtime_is_explicit(fixture, model):
    deploy, catalog, service = _inputs(fixture)
    with pytest.raises(UnsupportedRuntimeError) as exc:
        convert_release(
            deploy,
            catalog,
            service=service,
            model=model,
            version="v1",
        )
    message = str(exc.value)
    assert "unsupported in phase 1" in message
    assert "docs/integrations/integrate-with-model-deploy.md" in message
    assert "docs/tutorials/using-custom-backends.md" in message


def test_missing_version_errors():
    deploy, catalog, service = _inputs("vllm-standard")
    with pytest.raises(ConvertError, match="version is required"):
        convert_release(
            deploy, catalog, service=service, model="vllm-standard", version=""
        )


def test_plan_text_lists_mapping_without_api_fields():
    deploy, catalog, service = _inputs("vllm-standard")
    release = convert_release(
        deploy,
        catalog,
        service=service,
        model="vllm-standard",
        version="v1",
        worker="gpu-worker-a",
    )
    text = format_plan(release)
    assert "no API call" in text
    assert "vllm-standard" in text
    assert "/data/models/vllm-standard/v1" in text
    assert "vLLM" in text
    assert "HF_HUB_OFFLINE" in text
    assert "gpu-worker-a:cuda:0" in text
    assert "probe.path" in text


def test_plan_cli_stdout(capsys):
    deploy = FIXTURES / "vllm-standard" / "deploy.json"
    catalog = FIXTURES / "vllm-standard" / "model-catalog.json"
    service = FIXTURES / "vllm-standard" / "service.json"
    code = plan_main(
        [
            "--deploy",
            str(deploy),
            "--catalog",
            str(catalog),
            "--service",
            str(service),
            "--model",
            "vllm-standard",
            "--version",
            "v1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "local_path" in out
    assert "vLLM" in out


def test_convert_cli_yaml(tmp_path):
    output = tmp_path / "deploy.yaml"
    code = convert_main(
        [
            "--deploy",
            str(FIXTURES / "vllm-standard" / "deploy.json"),
            "--catalog",
            str(FIXTURES / "vllm-standard" / "model-catalog.json"),
            "--service",
            str(FIXTURES / "vllm-standard" / "service.json"),
            "--model",
            "vllm-standard",
            "--version",
            "v1",
            "--format",
            "yaml",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    doc = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert doc["source"] == "local_path"
    assert doc["backend"] == "vLLM"


def test_example_bridge_matches_schema_contract():
    example = (
        Path(__file__).resolve().parents[2]
        / "contrib"
        / "model-deploy"
        / "examples"
        / "vllm-bridge.json"
    )
    release = json.loads(example.read_text(encoding="utf-8"))
    assert required_release_fields(release) == []
    assert release["model_create"]["source"] == "local_path"
    assert apply_body(release)["backend"] == "vLLM"
