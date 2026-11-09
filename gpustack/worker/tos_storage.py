"""Immutable TOS model releases: manifest.json plus content-addressed objects.

Layout matches model-deploy (manifest + objects/<sha256>/<relative path>).
Credentials come from the assigned TosConnection (inline keys) or, when
credential_mode=env, from TOS_ACCESS_KEY / TOS_SECRET_KEY / TOS_SECURITY_TOKEN.
Secrets are never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Dict, List, Optional, Protocol
from urllib.parse import urlsplit

from tqdm import tqdm

from gpustack.schemas.models import ModelSource
from gpustack.utils.hub import FileEntry

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
LOCAL_COMPLETE_NAME = ".deployment-manifest.json"
_TEMP = ".tos-tmp"
_MAX_MANIFEST = 16 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_RESERVED = re.compile(
    r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I
)

ProgressCallback = Callable[[str, int], None]


class TosStorageError(RuntimeError):
    """A transfer failed without exposing SDK request credentials."""


class TosObjectClient(Protocol):
    def get_object_bytes(self, bucket: str, key: str) -> bytes: ...

    def try_get_object_bytes(self, bucket: str, key: str) -> Optional[bytes]: ...

    def download_object(self, bucket: str, key: str, dest: Path) -> None: ...

    def object_exists(self, bucket: str, key: str) -> bool: ...

    def upload_file(self, bucket: str, key: str, path: Path) -> None: ...

    def put_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        forbid_overwrite: bool = False,
    ) -> None: ...

    def list_keys(self, bucket: str, prefix: str) -> List[str]: ...


@dataclass(frozen=True)
class TosLocation:
    endpoint: str
    region: str
    bucket: str
    model: str
    version: str
    release_path: str

    @property
    def manifest_key(self) -> str:
        return f"{self.release_path}/{MANIFEST_NAME}"

    @property
    def uri(self) -> str:
        return f"tos://{self.bucket}/{self.release_path}"


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TosStorageError("TOS file path must be a safe relative path")
    parts = value.split("/")
    if any(
        p in ("", ".", "..")
        or p.endswith((".", " "))
        or _WINDOWS_RESERVED.fullmatch(p)
        or any(ord(c) < 32 or c in ':<>"|?*' for c in p)
        for p in parts
    ):
        raise TosStorageError("TOS file path is unsafe")
    if parts[0] in {MANIFEST_NAME, LOCAL_COMPLETE_NAME, _TEMP}:
        raise TosStorageError("TOS file path collides with release metadata")
    return value


def _require_id(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _ID.fullmatch(value)
        or value.endswith(".")
        or _WINDOWS_RESERVED.fullmatch(value)
    ):
        raise TosStorageError(
            f"{name} must start with an alphanumeric character and use "
            "only letters, digits, '.', '_' or '-'"
        )
    return value


def _require_model_id(name: str, value: Any) -> str:
    """Allow ``org/name`` style ids as a short path of ``_require_id`` segments."""
    if not isinstance(value, str) or not value:
        raise TosStorageError(f"{name} is required")
    parts = value.split("/")
    if len(parts) > 4:
        raise TosStorageError(f"{name} has too many path segments")
    return "/".join(_require_id(name, part) for part in parts)


def _https_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise TosStorageError("tos_endpoint is required")
    value = endpoint.strip()
    if "://" not in value:
        value = "https://" + value
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise TosStorageError(
            "tos_endpoint must be an HTTPS host without credentials or path"
        )
    return value.rstrip("/")


def resolve_release_path(
    model: str,
    version: str,
    release_path: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    """Resolve the bucket-relative release directory.

    Explicit ``release_path`` wins. Otherwise join ``prefix`` (or ``TOS_PREFIX``)
    with ``model/version``.
    """
    model = _require_model_id("tos_model", model)
    version = _require_id("tos_version", version)
    if release_path is not None and str(release_path).strip():
        raw = str(release_path).strip()
        try:
            for _, field, spec, conversion in Formatter().parse(raw):
                if field is not None and (
                    field not in {"model", "version"} or spec or conversion
                ):
                    raise ValueError()
            base = raw.format(model=model, version=version).strip("/")
        except (ValueError, KeyError, IndexError) as exc:
            raise TosStorageError(
                "tos_release_path only supports {model} and {version} placeholders"
            ) from exc
        suffix = f"{model}/{version}"
        if raw.endswith("/") and base != suffix and not base.endswith("/" + suffix):
            base = f"{base}/{suffix}"
        return _relative(base)

    env_prefix = prefix
    if env_prefix is None:
        env_prefix = os.environ.get("TOS_PREFIX", "")
    if not isinstance(env_prefix, str):
        raise TosStorageError("TOS prefix must be a string")
    pieces = [p for p in (env_prefix.strip("/"), model, version) if p]
    return _relative("/".join(pieces))


def location_from_source(model: ModelSource) -> TosLocation:
    endpoint = _https_endpoint(model.tos_endpoint or "")
    region = (model.tos_region or "").strip()
    bucket = (model.tos_bucket or "").strip()
    if not region or not bucket:
        raise TosStorageError("tos_region and tos_bucket are required")
    name = model.tos_model or ""
    version = model.tos_version or ""
    release_path = resolve_release_path(name, version, model.tos_release_path)
    return TosLocation(
        endpoint=endpoint,
        region=region,
        bucket=bucket,
        model=name,
        version=version,
        release_path=release_path,
    )


def object_key_for(release_path: str, digest: str, relative: str) -> str:
    return f"{release_path}/objects/{digest}/{_relative(relative)}"


def validate_manifest(value: Any, location: TosLocation) -> Dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("model") != location.model
        or value.get("version") != location.version
    ):
        raise TosStorageError("remote manifest format or model version mismatch")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise TosStorageError("remote manifest has no model files")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise TosStorageError("remote manifest file entry is invalid")
        path = _relative(entry.get("path"))
        folded = path.casefold()
        if folded in seen:
            raise TosStorageError("manifest has duplicate or case-colliding paths")
        seen.add(folded)
        size, digest = entry.get("size"), entry.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            raise TosStorageError("manifest file size or SHA256 is invalid")
        if not _SHA.fullmatch(digest):
            raise TosStorageError("manifest file size or SHA256 is invalid")
        expected = object_key_for(location.release_path, digest, path)
        if entry.get("object_key") != expected:
            raise TosStorageError("manifest object path is outside this release")
    for path in seen:
        pieces = path.split("/")
        if any("/".join(pieces[:i]) in seen for i in range(1, len(pieces))):
            raise TosStorageError("manifest has colliding file and directory paths")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    target = root.joinpath(*_relative(relative).split("/"))
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise TosStorageError("model path must not contain symlinks")
    if not target.resolve().is_relative_to(root.resolve()):
        raise TosStorageError("model file escaped the destination directory")
    return target


def cache_dir_for(model: ModelSource, cache_dir: Optional[str]) -> Path:
    location = location_from_source(model)
    root = os.environ.get("TOS_MODEL_ROOT", "").strip()
    if root:
        return Path(root) / location.model / location.version
    base = Path(cache_dir or "cache")
    return base / "tos" / location.bucket / location.release_path


def manifest_file_entries(manifest: Dict[str, Any]) -> List[FileEntry]:
    return [
        FileEntry(entry["path"], entry["size"], sha256=entry["sha256"])
        for entry in manifest["files"]
    ]


class SdkTosClient:
    """Thin wrapper around tos.TosClientV2. Import is deferred."""

    def __init__(
        self,
        endpoint: str,
        region: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        security_token: Optional[str] = None,
    ):
        ak = access_key if access_key is not None else os.environ.get("TOS_ACCESS_KEY")
        sk = secret_key if secret_key is not None else os.environ.get("TOS_SECRET_KEY")
        token = (
            security_token
            if security_token is not None
            else os.environ.get("TOS_SECURITY_TOKEN")
        )
        if not ak or not sk:
            raise TosStorageError(
                "TOS credentials are missing: store them on the TosConnection "
                "or set TOS_ACCESS_KEY and TOS_SECRET_KEY when credential_mode=env"
            )
        try:
            import tos
        except ImportError as exc:
            raise TosStorageError(
                "the tos SDK is not installed; pip install tos on the server "
                "that ingests releases and on workers that pull TOS models"
            ) from exc
        logging.getLogger("tos").setLevel(logging.CRITICAL)
        try:
            self._client = tos.TosClientV2(
                ak,
                sk,
                endpoint,
                region,
                security_token=token or None,
                max_retry_count=3,
                enable_crc=True,
                enable_verify_ssl=True,
            )
        except Exception as exc:
            raise TosStorageError("failed to initialize the TOS client") from exc

    def _read_object(
        self, bucket: str, key: str, *, allow_missing: bool
    ) -> Optional[bytes]:
        try:
            response = self._client.get_object(bucket, key)
            try:
                raw = response.read(_MAX_MANIFEST + 1)
            finally:
                closer = getattr(response, "close", None)
                if closer:
                    closer()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if allow_missing and status == 404:
                return None
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to read TOS object{suffix}") from None
        if len(raw) > _MAX_MANIFEST:
            raise TosStorageError("object exceeded the 16 MiB in-memory limit")
        return raw

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        raw = self._read_object(bucket, key, allow_missing=False)
        if raw is None:
            raise TosStorageError("failed to read TOS object")
        return raw

    def try_get_object_bytes(self, bucket: str, key: str) -> Optional[bytes]:
        return self._read_object(bucket, key, allow_missing=True)

    def download_object(self, bucket: str, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(bucket, key, str(dest))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to download TOS object{suffix}") from None

    def object_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(bucket, key)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return False
            status = getattr(exc, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to inspect TOS object{suffix}") from None
        return True

    def upload_file(self, bucket: str, key: str, path: Path) -> None:
        try:
            self._client.upload_file(bucket, key, str(path))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to upload TOS object{suffix}") from None

    def put_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        forbid_overwrite: bool = False,
    ) -> None:
        try:
            self._client.put_object(
                bucket,
                key,
                content=content,
                content_type=content_type,
                forbid_overwrite=forbid_overwrite,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to write TOS object{suffix}") from None

    def list_keys(self, bucket: str, prefix: str) -> List[str]:
        keys: List[str] = []
        continuation = ""
        try:
            while True:
                kwargs = {"prefix": prefix or ""}
                if continuation:
                    kwargs["continuation_token"] = continuation
                response = self._client.list_objects_type2(bucket, **kwargs)
                contents = getattr(response, "contents", None) or []
                for item in contents:
                    key = getattr(item, "key", None)
                    if key:
                        keys.append(key)
                truncated = bool(getattr(response, "is_truncated", False))
                continuation = getattr(response, "next_continuation_token", "") or ""
                if not truncated or not continuation:
                    break
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise TosStorageError(f"failed to list TOS objects{suffix}") from None
        return keys


def _parse_manifest(raw: bytes, location: TosLocation) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise TosStorageError("remote manifest is not valid JSON") from exc
    return validate_manifest(payload, location)


def fetch_manifest(location: TosLocation, client: TosObjectClient) -> Dict[str, Any]:
    return _parse_manifest(
        client.get_object_bytes(location.bucket, location.manifest_key), location
    )


def try_fetch_manifest(
    location: TosLocation, client: TosObjectClient
) -> Optional[Dict[str, Any]]:
    raw = client.try_get_object_bytes(location.bucket, location.manifest_key)
    if raw is None:
        return None
    return _parse_manifest(raw, location)


def manifest_bytes(manifest: Dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_total_size(manifest: Dict[str, Any]) -> int:
    return sum(int(entry["size"]) for entry in manifest.get("files") or [])


def _iter_local_files(root: Path, relatives: Optional[List[str]]) -> List[Path]:
    if relatives:
        selected = []
        for rel in relatives:
            path = _safe_path(root, rel)
            if not path.is_file():
                raise TosStorageError(f"archive path is not a file: {rel}")
            selected.append(path)
        return selected
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TosStorageError("local model directory must not contain symlinks")
        if not path.is_file():
            continue
        files.append(path)
    return files


def scan_local_release(
    local_dir: Path,
    location: TosLocation,
    relatives: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a model-deploy manifest from a local directory or hub snapshot."""
    root = Path(local_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise TosStorageError("local model path must be an existing directory")
    files = []
    for path in _iter_local_files(root, relatives):
        relative = path.relative_to(root).as_posix()
        if relative.split("/")[0] in {MANIFEST_NAME, LOCAL_COMPLETE_NAME, _TEMP}:
            continue
        relative = _relative(relative)
        digest = sha256_file(path)
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest,
                "object_key": object_key_for(location.release_path, digest, relative),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "model": location.model,
        "version": location.version,
        "files": files,
    }
    if extra:
        payload.update(extra)
    return validate_manifest(payload, location)


def upload_release(
    location: TosLocation,
    local_dir: Path,
    client: TosObjectClient,
    relatives: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Publish an immutable TOS release. Same version + different bytes is rejected.

    Objects are uploaded first. The manifest is written last with
    ``forbid_overwrite`` so a crash mid-upload leaves no catalog entry and a
    retry can resume by skipping objects that already exist.
    """
    root = Path(local_dir).absolute()
    manifest = scan_local_release(root, location, relatives=relatives, extra=extra)
    raw = manifest_bytes(manifest)
    if len(raw) > _MAX_MANIFEST:
        raise TosStorageError("manifest exceeded the 16 MiB limit")
    previous = try_fetch_manifest(location, client)
    if previous is not None:
        if previous != manifest:
            raise TosStorageError(
                "this model version is already published with different files; "
                "use a new tos_version"
            )
        if progress:
            total = manifest_total_size(manifest)
            progress(total, total)
        return previous
    total = manifest_total_size(manifest)
    done = 0
    for entry in manifest["files"]:
        if not client.object_exists(location.bucket, entry["object_key"]):
            client.upload_file(
                location.bucket, entry["object_key"], _safe_path(root, entry["path"])
            )
        done += entry["size"]
        if progress:
            progress(done, total)
    try:
        client.put_bytes(
            location.bucket,
            location.manifest_key,
            raw,
            content_type="application/json",
            forbid_overwrite=True,
        )
    except TosStorageError:
        actual = try_fetch_manifest(location, client)
        if actual != manifest:
            raise TosStorageError(
                "failed to publish manifest, or another writer took this version"
            ) from None
    return manifest


def list_releases(
    client: TosObjectClient, bucket: str, prefix: str
) -> List[Dict[str, str]]:
    """List immutable releases whose manifest sits under ``prefix``."""
    from gpustack.utils.tos_scope import object_key_in_prefix

    keys = client.list_keys(bucket, prefix or "")
    found: List[Dict[str, str]] = []
    seen = set()
    marker = f"/{MANIFEST_NAME}"
    for key in keys:
        if not key.endswith(marker) and key != MANIFEST_NAME:
            continue
        if not object_key_in_prefix(prefix, key):
            continue
        rel = key[: -len(MANIFEST_NAME)].strip("/")
        if prefix:
            trimmed = prefix.strip("/")
            if rel == trimmed:
                continue
            if rel.startswith(trimmed + "/"):
                rel = rel[len(trimmed) + 1 :]
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        version = parts[-1]
        model = "/".join(parts[:-1])
        identity = (model, version)
        if identity in seen:
            continue
        seen.add(identity)
        found.append({"tos_model": model, "tos_version": version})
    return found


def _file_matches(path: Path, entry: Dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and sha256_file(path) == entry["sha256"]
    )


def download_release(
    location: TosLocation,
    destination: Path,
    client: TosObjectClient,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[str]:
    """Download and SHA256-verify every manifest object into destination.

    Returns the destination directory as a one-element path list so backends
    start from the local tree the same way a Hub snapshot does.
    """
    manifest = fetch_manifest(location, client)
    root = destination.absolute()
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise TosStorageError("TOS destination must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    complete = root / LOCAL_COMPLETE_NAME
    if complete.is_file() and not complete.is_symlink():
        try:
            saved = json.loads(complete.read_bytes())
        except (ValueError, OSError):
            saved = None
        if saved == manifest and all(
            _file_matches(_safe_path(root, entry["path"]), entry)
            for entry in manifest["files"]
        ):
            return [str(root)]

    temp = root / _TEMP
    temp.mkdir(exist_ok=True)
    for entry in manifest["files"]:
        target = _safe_path(root, entry["path"])
        if _file_matches(target, entry):
            if progress_callback:
                progress_callback(entry["path"], entry["size"])
            continue
        partial = temp / f"{hashlib.sha256(entry['path'].encode()).hexdigest()}.part"
        if partial.exists() or partial.is_symlink():
            partial.unlink()
        client.download_object(location.bucket, entry["object_key"], partial)
        if not partial.is_file() or partial.stat().st_size != entry["size"]:
            partial.unlink(missing_ok=True)
            raise TosStorageError(f"TOS object size mismatch: {entry['path']}")
        digest = sha256_file(partial)
        if digest != entry["sha256"]:
            partial.unlink(missing_ok=True)
            raise TosStorageError(f"TOS object SHA256 mismatch: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, target)
        with tqdm(
            total=entry["size"],
            unit="B",
            unit_scale=True,
            desc=entry["path"],
        ) as bar:
            bar.update(entry["size"])
        if progress_callback:
            progress_callback(entry["path"], entry["size"])

    complete.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return [str(root)]
