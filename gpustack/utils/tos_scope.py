"""Prefix, path, and endpoint guards for TOS connections and archives."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from gpustack.api.exceptions import InvalidException


def normalize_prefix(prefix: Optional[str]) -> str:
    """Return a relative prefix without a leading slash.

    Empty prefix means the bucket root. ``..`` and absolute paths are rejected.
    """
    if prefix is None:
        return ""
    value = prefix.strip().replace("\\", "/")
    if not value:
        return ""
    if value.startswith("/"):
        raise InvalidException(message="TOS prefix must be relative to the bucket")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise InvalidException(message="TOS prefix must not contain '..'")
    return "/".join(parts)


def https_endpoint(endpoint: str) -> str:
    from gpustack.worker.tos_storage import TosStorageError, _https_endpoint

    try:
        return _https_endpoint(endpoint)
    except TosStorageError as exc:
        raise InvalidException(message=str(exc)) from exc


def env_endpoint_allows(connection_endpoint: str) -> bool:
    """Env-mode keys may only be sent to TOS_ENDPOINT / TOS_ALLOWED_ENDPOINTS."""
    pinned = os.environ.get("TOS_ENDPOINT", "").strip()
    allowed = os.environ.get("TOS_ALLOWED_ENDPOINTS", "").strip()
    candidates = []
    if pinned:
        candidates.append(pinned)
    if allowed:
        candidates.extend(part.strip() for part in allowed.split(",") if part.strip())
    if not candidates:
        return False
    target = https_endpoint(connection_endpoint)
    for item in candidates:
        try:
            if https_endpoint(item) == target:
                return True
        except InvalidException:
            continue
    return False


def object_key_in_prefix(prefix: str, key: str) -> bool:
    """True when ``key`` is exactly under the connection prefix."""
    normalized_prefix = normalize_prefix(prefix)
    normalized_key = key.lstrip("/")
    if any(part == ".." for part in normalized_key.split("/")):
        return False
    if not normalized_prefix:
        return ".." not in normalized_key.split("/")
    return normalized_key == normalized_prefix or normalized_key.startswith(
        normalized_prefix + "/"
    )


def artifact_uri(bucket: str, prefix: str, model: str, version: str) -> str:
    root = normalize_prefix(prefix)
    model_part = model.strip().strip("/")
    version_part = version.strip().strip("/")
    path = "/".join(p for p in (root, model_part, version_part) if p)
    return f"tos://{bucket}/{path}"


def safe_relative(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; reject traversal and symlink escape."""
    rel = relative.replace("\\", "/").lstrip("/")
    if not rel or rel.startswith("/") or any(p == ".." for p in rel.split("/")):
        raise InvalidException(message=f"Illegal archive path: {relative}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise InvalidException(
            message=f"Archive path escapes the model cache: {relative}"
        ) from exc
    if candidate.is_symlink() or any(
        parent.is_symlink() for parent in candidate.parents if root in parent.parents
    ):
        target = candidate.resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise InvalidException(
                message=f"Archive path symlink escapes the model cache: {relative}"
            ) from exc
    return candidate


def path_is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def looks_gguf(paths: list[str]) -> bool:
    return any(p.lower().endswith(".gguf") for p in paths)


def disk_full(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno == getattr(os, "ENOSPC", 28):
        return True
    text = str(exc).lower()
    return "no space" in text or "enospc" in text


def classify_archive_error(exc: BaseException) -> str:
    from gpustack.schemas.tos import TosArchiveErrorCode

    if disk_full(exc):
        return TosArchiveErrorCode.DISK_FULL.value
    message = str(exc).lower()
    if "already published" in message or "another writer" in message:
        return TosArchiveErrorCode.IMMUTABLE_CONFLICT.value
    if "sha256" in message or "size mismatch" in message:
        return TosArchiveErrorCode.CHECKSUM_MISMATCH.value
    if any(
        token in message
        for token in ("credential", "access_key", "http 403", "http 401")
    ):
        return TosArchiveErrorCode.INVALID_CREDENTIALS.value
    if "symlink" in message or "escaped" in message or "unsafe" in message:
        return TosArchiveErrorCode.PATH_ESCAPE.value
    if "prefix" in message:
        return TosArchiveErrorCode.PREFIX_DENIED.value
    return TosArchiveErrorCode.INCOMPLETE_UPLOAD.value
