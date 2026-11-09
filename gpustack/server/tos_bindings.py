"""Bind TosConnection coordinates onto Model / ModelFile payloads."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext, assert_resource_visible
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.tos import TosArchiveErrorCode, TosConnection
from gpustack.utils.tos_credentials import (
    access_from_connection,
    assert_connection_owned_by,
    client_from_access,
)
from gpustack.utils.tos_secrets import TosSecretError, open_secret, seal_secret
from gpustack.worker.tos_storage import resolve_release_path
from gpustack.utils.tos_scope import (
    enforce_endpoint_trust,
    https_endpoint,
    normalize_prefix,
)


def redact_access_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if len(value) <= 6:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def public_access_key(stored: Optional[str]) -> Optional[str]:
    if not stored:
        return None
    try:
        return redact_access_key(open_secret(stored))
    except TosSecretError:
        return None


def connection_scope(prefix: Optional[str]) -> str:
    return "bucket" if not normalize_prefix(prefix) else "prefix"


async def load_visible_connection(
    session, ctx: TenantContext, connection_id: int
) -> TosConnection:
    connection = await TosConnection.one_by_id(session, connection_id)
    if not connection or connection.deleted_at is not None:
        raise NotFoundException(message=f"TOS connection {connection_id} not found")
    assert_resource_visible(
        ctx,
        connection,
        not_found_message=f"TOS connection {connection_id} not found",
    )
    return connection


def apply_connection_coordinates(payload: Any, connection: TosConnection) -> None:
    payload.tos_connection_id = connection.id
    payload.tos_endpoint = https_endpoint(connection.endpoint)
    payload.tos_region = connection.region
    payload.tos_bucket = connection.bucket
    prefix = normalize_prefix(connection.prefix)
    payload.tos_release_path = resolve_release_path(
        payload.tos_model,
        payload.tos_version,
        prefix=prefix,
    )


async def bind_tos_connection(
    session,
    ctx: TenantContext,
    payload: Any,
    *,
    expected_owner: Optional[int] = None,
) -> None:
    if getattr(payload, "source", None) != SourceEnum.TOS:
        return
    connection_id = getattr(payload, "tos_connection_id", None)
    if not connection_id:
        if payload.tos_endpoint:
            payload.tos_endpoint = enforce_endpoint_trust(
                payload.tos_endpoint, credential_mode="env"
            )
        return
    connection = await load_visible_connection(session, ctx, connection_id)
    if expected_owner is not None:
        assert_connection_owned_by(connection, expected_owner)
    apply_connection_coordinates(payload, connection)


async def bind_auto_archive_connection(
    session,
    ctx: TenantContext,
    payload: Any,
    *,
    expected_owner: Optional[int] = None,
) -> None:
    spec = auto_archive_spec(payload)
    if not spec or not spec.get("enabled"):
        return
    connection_id = spec.get("connection_id")
    if not connection_id:
        raise InvalidException(message="auto_archive requires connection_id")
    if not spec.get("tos_model") or not spec.get("tos_version"):
        raise InvalidException(
            message="auto_archive requires tos_model and tos_version"
        )
    connection = await load_visible_connection(session, ctx, connection_id)
    owner = expected_owner
    if owner is None:
        owner = getattr(payload, "owner_principal_id", None) or ctx.current_principal_id
    assert_connection_owned_by(connection, owner)


def origin_from_model_file(model_file) -> dict:
    return {
        "origin_source": (
            model_file.source.value
            if hasattr(model_file.source, "value")
            else str(model_file.source)
        ),
        "origin_huggingface_repo_id": model_file.huggingface_repo_id,
        "origin_huggingface_filename": model_file.huggingface_filename,
        "origin_model_scope_model_id": model_file.model_scope_model_id,
        "origin_model_scope_file_path": model_file.model_scope_file_path,
        "origin_local_path": model_file.local_path,
        "origin_worker_id": model_file.worker_id,
        "origin_model_file_id": model_file.id,
    }


def auto_archive_spec(payload: Any) -> Optional[dict]:
    spec = getattr(payload, "auto_archive", None)
    if not spec:
        return None
    if isinstance(spec, dict):
        return spec
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    return None


def auto_archive_enabled(payload: Any) -> bool:
    spec = auto_archive_spec(payload)
    return bool(spec and spec.get("enabled"))


def seal_connection_secrets(payload: dict) -> dict:
    for key in ("access_key", "secret_key", "security_token"):
        if payload.get(key):
            payload[key] = seal_secret(payload[key])
    return payload


async def enqueue_archive_job(
    session,
    model_file,
    connection: TosConnection,
    tos_model: str,
    tos_version: str,
    paths=None,
    backend_hint=None,
):
    from sqlalchemy.exc import IntegrityError

    from gpustack.schemas.model_files import ModelFileStateEnum
    from gpustack.schemas.tos import (
        TosArchiveJob,
        TosArchiveStateEnum,
    )
    from gpustack.utils.tos_scope import artifact_uri, looks_gguf, normalize_prefix
    from gpustack.worker import tos_storage

    assert_connection_owned_by(connection, model_file.owner_principal_id)
    if paths:
        from gpustack.utils.model_file_listing import sanitize_archive_relative

        paths = [sanitize_archive_relative(path) for path in paths]
    if model_file.state != ModelFileStateEnum.READY:
        raise InvalidException(
            message="Only READY model files can be archived",
            details={"error_code": TosArchiveErrorCode.NOT_READY.value},
        )
    try:
        tos_storage.resolve_release_path(
            tos_model, tos_version, prefix=normalize_prefix(connection.prefix)
        )
    except tos_storage.TosStorageError as exc:
        raise InvalidException(message=str(exc)) from exc

    existing = await TosArchiveJob.one_by_fields(
        session,
        {
            "connection_id": connection.id,
            "tos_model": tos_model,
            "tos_version": tos_version,
        },
    )
    if existing and getattr(existing, "deleted_at", None) is None:
        if existing.state == TosArchiveStateEnum.READY.value:
            raise AlreadyExistsException(
                message="this connection already published that model version",
                details={
                    "error_code": TosArchiveErrorCode.IMMUTABLE_CONFLICT.value,
                    "retry_job_id": existing.id,
                },
            )
        if existing.state == TosArchiveStateEnum.ERROR.value:
            raise AlreadyExistsException(
                message="this version already failed; POST /v2/tos/archives/"
                f"{existing.id}/retry instead of creating a new job",
                details={
                    "error_code": TosArchiveErrorCode.VERSION_RETRY_REQUIRED.value,
                    "retry_job_id": existing.id,
                },
            )
        raise AlreadyExistsException(
            message="this connection already has that model version in progress",
            details={
                "error_code": TosArchiveErrorCode.CONCURRENT_VERSION.value,
                "retry_job_id": existing.id,
            },
        )

    provenance = origin_from_model_file(model_file)
    provenance["selected_paths"] = paths
    provenance["backend_hint"] = backend_hint
    provenance["gguf"] = looks_gguf(paths or model_file.resolved_paths or [])
    provenance["artifact_uri"] = artifact_uri(
        connection.bucket, connection.prefix, tos_model, tos_version
    )
    job = TosArchiveJob(
        model_file_id=model_file.id,
        connection_id=connection.id,
        worker_id=model_file.worker_id,
        tos_model=tos_model,
        tos_version=tos_version,
        paths=paths,
        state=TosArchiveStateEnum.QUEUED.value,
        provenance=provenance,
        cluster_id=model_file.cluster_id,
        owner_principal_id=model_file.owner_principal_id,
    )
    try:
        created = await TosArchiveJob.create(session, job, auto_commit=False)
    except IntegrityError as exc:
        raise AlreadyExistsException(
            message="this connection already has that model version; "
            "use a new tos_version or retry the existing job",
            details={"error_code": TosArchiveErrorCode.CONCURRENT_VERSION.value},
        ) from exc
    model_file.archive_state = created.state
    model_file.archive_error_code = None
    model_file.archive_error_message = None
    await model_file.update(session, auto_commit=False)
    return created


def origin_locked_provenance(stored: dict, incoming: dict) -> dict:
    origin = {key: stored[key] for key in stored if str(key).startswith("origin_")}
    merged = {**stored, **incoming, **origin}
    return merged


def verify_committed_manifest(connection, job, incoming: dict) -> dict:
    """Fetch the remote committed manifest and refuse a fake complete."""
    from gpustack.utils.tos_scope import looks_gguf, normalize_prefix
    from gpustack.worker import tos_storage

    assert_connection_owned_by(connection, job.owner_principal_id)
    access = access_from_connection(
        connection, tos_model=job.tos_model, tos_version=job.tos_version
    )
    client = client_from_access(access)
    prefix = normalize_prefix(connection.prefix)
    location = tos_storage.TosLocation(
        endpoint=access.endpoint,
        region=access.region,
        bucket=connection.bucket,
        model=job.tos_model,
        version=job.tos_version,
        release_path=tos_storage.resolve_release_path(
            job.tos_model, job.tos_version, prefix=prefix
        ),
    )
    try:
        remote = tos_storage.fetch_manifest(location, client)
    except tos_storage.TosStorageError as exc:
        raise InvalidException(
            message="committed TOS manifest is missing or unreadable",
            details={"error_code": TosArchiveErrorCode.MANIFEST_MISMATCH.value},
        ) from exc
    raw = tos_storage.manifest_bytes(remote)
    digest = hashlib.sha256(raw).hexdigest()
    claimed = incoming.get("manifest_sha256")
    if not claimed or claimed != digest:
        raise InvalidException(
            message="complete sha256 does not match the committed remote manifest",
            details={"error_code": TosArchiveErrorCode.CHECKSUM_MISMATCH.value},
        )
    if remote.get("model") != job.tos_model or remote.get("version") != job.tos_version:
        raise InvalidException(
            message="remote manifest model/version does not match the archive job",
            details={"error_code": TosArchiveErrorCode.MANIFEST_MISMATCH.value},
        )
    files = [entry["path"] for entry in remote.get("files") or []]
    if job.paths:
        missing = [path for path in job.paths if path not in files]
        if missing:
            raise InvalidException(
                message="committed manifest does not include the archived paths",
                details={"error_code": TosArchiveErrorCode.MANIFEST_MISMATCH.value},
            )
    stored = dict(job.provenance or {})
    incoming_safe = {
        "manifest_sha256": digest,
        "manifest_bytes": len(raw),
        "object_count": len(files),
        "artifact_uri": location.uri,
        "gguf": looks_gguf(files),
        "selected_paths": job.paths if job.paths is not None else files,
    }
    return origin_locked_provenance(stored, incoming_safe)


async def stamp_tos_provenance_from_manifest(
    session, model_in, owner_principal_id: Optional[int] = None
) -> None:
    """Overwrite client provenance with the trusted connection manifest."""
    from gpustack.worker import tos_storage

    if getattr(model_in, "source", None) != SourceEnum.TOS:
        return
    connection_id = getattr(model_in, "tos_connection_id", None)
    owner = owner_principal_id or getattr(model_in, "owner_principal_id", None)
    if connection_id:
        connection = await TosConnection.one_by_id(session, connection_id)
        assert_connection_owned_by(connection, owner)
        access = access_from_connection(
            connection,
            tos_model=model_in.tos_model,
            tos_version=model_in.tos_version,
        )
        prefix = normalize_prefix(connection.prefix)
        location = tos_storage.TosLocation(
            endpoint=access.endpoint,
            region=access.region,
            bucket=connection.bucket,
            model=model_in.tos_model,
            version=model_in.tos_version,
            release_path=tos_storage.resolve_release_path(
                model_in.tos_model, model_in.tos_version, prefix=prefix
            ),
        )
        client = client_from_access(access)
    else:
        from gpustack.utils.tos_credentials import access_from_legacy_source

        try:
            access = access_from_legacy_source(model_in)
        except tos_storage.TosStorageError as exc:
            raise InvalidException(message=str(exc)) from exc
        client = client_from_access(access)
        location = tos_storage.location_from_source(model_in)
    try:
        manifest = tos_storage.fetch_manifest(location, client)
    except tos_storage.TosStorageError as exc:
        raise InvalidException(
            message="TOS deploy requires a readable trusted manifest",
            details={"error_code": TosArchiveErrorCode.MANIFEST_MISMATCH.value},
        ) from exc
    from gpustack.worker.tos_storage import provenance_from_manifest

    stamped = provenance_from_manifest(manifest, location)
    current = dict(getattr(model_in, "provenance", None) or {})
    current.update(stamped)
    model_in.provenance = current
