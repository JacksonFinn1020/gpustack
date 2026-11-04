from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from apply import (
    ApplyError,
    apply_release,
    default_http_post,
    main as apply_main,
    redact,
)
from bridge import convert_release, load_json

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "model_deploy"


def _vllm_release(**kwargs):
    root = FIXTURES / "vllm-standard"
    return convert_release(
        load_json(root / "deploy.json"),
        load_json(root / "model-catalog.json"),
        service=load_json(root / "service.json"),
        model="vllm-standard",
        version="v1",
        **kwargs,
    )


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else str(payload))

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_dry_run_does_not_post():
    calls: List[Tuple] = []

    def boom(*_args, **_kwargs):
        calls.append(_args)
        raise AssertionError("dry-run must not POST")

    result = apply_release(
        _vllm_release(),
        "https://gpustack.example.com",
        "secret-token",
        dry_run=True,
        http_post=boom,
    )
    assert calls == []
    assert result["dry_run"] is True
    assert result["method"] == "POST"
    assert result["url"] == "https://gpustack.example.com/v2/models"
    assert result["body"]["source"] == "local_path"
    assert result["body"]["name"] == "vllm-standard"
    assert "cluster_name" not in result["body"]


def test_apply_posts_bearer_and_model_create_body():
    captured: Dict[str, Any] = {}

    def fake_post(url, headers, body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return 201, {"id": 42, "name": body["name"]}, '{"id":42}'

    result = apply_release(
        _vllm_release(cluster_name="prod"),
        "https://gpustack.example.com/",
        "secret-token",
        http_post=fake_post,
    )
    assert result["id"] == 42
    assert captured["url"] == "https://gpustack.example.com/v2/models"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["body"]["source"] == "local_path"
    assert captured["body"]["local_path"] == "/data/models/vllm-standard/v1"
    assert captured["body"]["backend"] == "vLLM"
    assert "cluster_name" not in captured["body"]


def test_non_2xx_redacts_secret():
    def fake_post(_url, _headers, _body, _timeout):
        return 401, None, "unauthorized secret-token Bearer secret-token"

    with pytest.raises(ApplyError) as exc:
        apply_release(
            _vllm_release(),
            "https://gpustack.example.com",
            "secret-token",
            http_post=fake_post,
        )
    message = str(exc.value)
    assert "secret-token" not in message
    assert "401" in message
    assert "***" in message


def test_transport_error_is_redacted():
    def fake_post(*_args, **_kwargs):
        raise ConnectionError("failed to speak to host with secret-token")

    with pytest.raises(ApplyError) as exc:
        apply_release(
            _vllm_release(),
            "https://gpustack.example.com",
            "secret-token",
            http_post=fake_post,
        )
    assert "secret-token" not in str(exc.value)


def test_redact_bearer_without_knowing_secret():
    assert "leak" not in redact("Authorization: Bearer leak", None)


def test_missing_credentials():
    with pytest.raises(ApplyError, match="GPUSTACK_SERVER_URL"):
        apply_release(_vllm_release(), "", "key", dry_run=True)
    with pytest.raises(ApplyError, match="GPUSTACK_API_KEY"):
        apply_release(_vllm_release(), "https://gpustack.example.com", "")


def test_apply_cli_dry_run(capsys, monkeypatch):
    monkeypatch.delenv("GPUSTACK_API_KEY", raising=False)
    code = apply_main(
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
            "--server-url",
            "https://gpustack.example.com",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "dry_run" in out
    assert "/v2/models" in out


def test_default_http_post_uses_httpx(monkeypatch):
    posted: Dict[str, Any] = {}

    def fake_post(url, headers, json, timeout):
        posted.update(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _FakeResponse(200, {"ok": True}, '{"ok":true}')

    fake_httpx = MagicMock()
    fake_httpx.post = fake_post
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)
    status, payload, text = default_http_post(
        "https://gpustack.example.com/v2/models",
        {"Authorization": "Bearer x"},
        {"name": "m"},
        5.0,
    )
    assert status == 200
    assert payload == {"ok": True}
    assert posted["url"].endswith("/v2/models")
    assert posted["headers"]["Authorization"] == "Bearer x"
    assert text
