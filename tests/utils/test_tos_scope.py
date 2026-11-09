import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.server.tos_bindings import redact_access_key
from gpustack.utils.tos_scope import classify_archive_error
from gpustack.utils.tos_scope import (
    artifact_uri,
    looks_gguf,
    normalize_prefix,
    object_key_in_prefix,
)
from gpustack.worker.tos_storage import TosStorageError


def test_normalize_prefix_rejects_traversal():
    assert normalize_prefix("org-a/models") == "org-a/models"
    assert normalize_prefix("") == ""
    with pytest.raises(InvalidException):
        normalize_prefix("../etc")
    with pytest.raises(InvalidException):
        normalize_prefix("/abs")


def test_object_key_in_prefix():
    assert object_key_in_prefix("org-a/models", "org-a/models/qwen3/v1/manifest.json")
    assert not object_key_in_prefix(
        "org-a/models", "org-b/models/qwen3/v1/manifest.json"
    )
    assert not object_key_in_prefix("org-a/models", "org-a/models/../escape")


def test_redact_access_key():
    assert redact_access_key("AKLTSECRET99") == "AK***99"
    assert redact_access_key("ab") == "***"


def test_gguf_and_artifact_uri():
    assert looks_gguf(["model.GGUF", "config.json"])
    assert not looks_gguf(["config.json"])
    assert (
        artifact_uri("bucket", "org-a/models", "qwen3", "v1")
        == "tos://bucket/org-a/models/qwen3/v1"
    )


def test_classify_immutable_and_disk():
    assert classify_archive_error(TosStorageError("already published")) == (
        "immutable_conflict"
    )
    err = OSError(28, "No space left on device")
    err.errno = 28
    assert classify_archive_error(err) == "disk_full"


def test_classify_connection_refused_is_worker_unreachable():
    assert (
        classify_archive_error(ConnectionRefusedError("Connection refused"))
        == "worker_unreachable"
    )
    refused = OSError(111, "Connection refused")
    refused.errno = 111
    assert classify_archive_error(refused) == "worker_unreachable"
