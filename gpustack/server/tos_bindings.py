"""Bind TosConnection coordinates onto Model / ModelFile payloads."""

from __future__ import annotations

from typing import Any, Optional

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext, assert_resource_visible
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.tos import TosConnection
from gpustack.worker.tos_storage import resolve_release_path
from gpustack.utils.tos_scope import https_endpoint, normalize_prefix


def redact_access_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if len(value) <= 6:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


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


async def bind_tos_connection(session, ctx: TenantContext, payload: Any) -> None:
    if getattr(payload, "source", None) != SourceEnum.TOS:
        return
    connection_id = getattr(payload, "tos_connection_id", None)
    if not connection_id:
        if payload.tos_endpoint:
            payload.tos_endpoint = https_endpoint(payload.tos_endpoint)
        return
    connection = await load_visible_connection(session, ctx, connection_id)
    apply_connection_coordinates(payload, connection)


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

    from gpustack.api.exceptions import AlreadyExistsException, InvalidException
    from gpustack.schemas.model_files import ModelFileStateEnum
    from gpustack.schemas.tos import (
        TosArchiveErrorCode,
        TosArchiveJob,
        TosArchiveStateEnum,
    )
    from gpustack.utils.tos_scope import artifact_uri, looks_gguf, normalize_prefix
    from gpustack.worker import tos_storage

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
