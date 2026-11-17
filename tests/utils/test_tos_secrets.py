import pytest

from gpustack.utils import tos_secrets
from gpustack.utils.tos_secrets import TosSecretError


def test_seal_round_trip_with_server_key(monkeypatch):
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "unit-test-tos-key")
    sealed = tos_secrets.seal_secret("super-secret")
    assert sealed.startswith("gsk1:")
    assert "super-secret" not in sealed
    assert tos_secrets.open_secret(sealed) == "super-secret"


def test_seal_fails_closed_without_dedicated_key(monkeypatch, caplog):
    monkeypatch.delenv("GPUSTACK_TOS_SECRET_KEY", raising=False)
    monkeypatch.setenv("GPUSTACK_JWT_SECRET_KEY", "jwt-must-not-seal-tos")
    with pytest.raises(TosSecretError, match="GPUSTACK_TOS_SECRET_KEY"):
        tos_secrets.seal_secret("super-secret")
    assert "plaintext" not in caplog.text.lower()


def test_plaintext_open_is_rejected():
    with pytest.raises(TosSecretError, match="not sealed"):
        tos_secrets.open_secret("legacy-plain")
    assert tos_secrets.open_secret(None) is None


def test_previous_key_decrypts_rotated_rows(monkeypatch):
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "old-key")
    sealed = tos_secrets.seal_secret("super-secret")
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "new-key")
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY_PREVIOUS", "old-key")
    assert tos_secrets.open_secret(sealed) == "super-secret"
    rotated = tos_secrets.seal_secret("super-secret")
    monkeypatch.delenv("GPUSTACK_TOS_SECRET_KEY_PREVIOUS")
    assert tos_secrets.open_secret(rotated) == "super-secret"
