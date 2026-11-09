from gpustack.utils import tos_secrets


def test_seal_round_trip_with_server_key(monkeypatch):
    monkeypatch.setenv("GPUSTACK_TOS_SECRET_KEY", "unit-test-tos-key")
    sealed = tos_secrets.seal_secret("super-secret")
    assert sealed.startswith("gsk1:")
    assert "super-secret" not in sealed
    assert tos_secrets.open_secret(sealed) == "super-secret"


def test_unprefixed_secret_is_returned_as_stored():
    assert tos_secrets.open_secret("legacy-plain") == "legacy-plain"
    assert tos_secrets.open_secret(None) is None
