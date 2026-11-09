from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.streaming import tenant_streaming
from gpustack.api.tenant import (
    TenantContext,
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.tos import (
    TosArchiveCompleteRequest,
    TosArchiveErrorCode,
    TosArchiveJob,
    TosArchiveJobCreate,
    TosArchiveJobListParams,
    TosArchiveJobPublic,
    TosArchiveJobUpdate,
    TosArchiveJobsPublic,
    TosArchiveStateEnum,
    TosConnection,
    TosReleaseListItem,
    TosReleasePublic,
    TosReleaseFilePublic,
    TosTransport,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.tos_bindings import (
    load_visible_connection,
)
from gpustack.utils.tos_scope import (
    artifact_uri,
    classify_archive_error,
    looks_gguf,
    normalize_prefix,
    object_key_in_prefix,
)
from gpustack.worker import tos_storage

router = APIRouter()


def _worker_id(ctx: TenantContext) -> Optional[int]:
    worker = getattr(ctx.user, "worker", None)
    return getattr(worker, "id", None)


def _require_assigned_worker(ctx: TenantContext, job: TosArchiveJob) -> None:
    if ctx.user is None or ctx.user.kind != PrincipalType.SYSTEM:
        raise ForbiddenException(message="Only the assigned worker may call this")
    if _worker_id(ctx) != job.worker_id:
        raise ForbiddenException(message="Archive job is assigned to another worker")


def _sync_model_file_archive(model_file: ModelFile, job: TosArchiveJob) -> None:
    model_file.archive_state = job.state
    model_file.archive_error_code = job.error_code
    model_file.archive_error_message = job.state_message
    if job.state == TosArchiveStateEnum.READY.value:
        provenance = dict(model_file.provenance or {})
        provenance.update(job.provenance or {})
        model_file.provenance = provenance


@router.get("/archives", response_model=TosArchiveJobsPublic)
async def list_archives(
    ctx: TenantContextDep,
    params: TosArchiveJobListParams = Depends(),
    worker_id: int = None,
    model_file_id: int = None,
    state: str = None,
):
    fields = {}
    if worker_id:
        fields["worker_id"] = worker_id
    if model_file_id:
        fields["model_file_id"] = model_file_id
    if state:
        fields["state"] = state
    if params.watch:
        return StreamingResponse(
            tenant_streaming(TosArchiveJob, ctx, fields=fields),
            media_type="text/event-stream",
        )
    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, TosArchiveJob)
        return await TosArchiveJob.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.get("/archives/{id}", response_model=TosArchiveJobPublic)
async def get_archive(session: SessionDep, ctx: TenantContextDep, id: int):
    job = await TosArchiveJob.one_by_id(session, id)
    assert_resource_visible(ctx, job, not_found_message=f"Archive job {id} not found")
    return job


@router.post("/archives", response_model=TosArchiveJobPublic)
async def create_archive(
    session: SessionDep, ctx: TenantContextDep, input: TosArchiveJobCreate
):
    model_file = await ModelFile.one_by_id(session, input.model_file_id)
    assert_resource_visible(
        ctx,
        model_file,
        not_found_message=f"Model file {input.model_file_id} not found",
    )
    if not model_file.worker_id:
        raise InvalidException(message="Model file has no assigned worker")
    if not model_file.resolved_paths:
        raise InvalidException(message="READY model file has no resolved local path")
    connection = await load_visible_connection(session, ctx, input.connection_id)
    if (
        model_file.owner_principal_id
        and connection.owner_principal_id
        and model_file.owner_principal_id != connection.owner_principal_id
    ):
        raise ForbiddenException(
            message="TOS connection is not in the same organization as the model file"
        )
    from gpustack.server.tos_bindings import enqueue_archive_job

    try:
        created = await enqueue_archive_job(
            session,
            model_file,
            connection,
            input.tos_model,
            input.tos_version,
            paths=input.paths,
            backend_hint=input.backend_hint,
        )
        await session.commit()
    except (InvalidException, AlreadyExistsException, ForbiddenException):
        raise
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create archive job: {e}"
        ) from e
    return created


@router.put("/archives/{id}", response_model=TosArchiveJobPublic)
async def update_archive(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: TosArchiveJobUpdate,
):
    job = await TosArchiveJob.one_by_id(session, id)
    assert_resource_visible(ctx, job, not_found_message=f"Archive job {id} not found")
    _require_assigned_worker(ctx, job)
    if job.state == TosArchiveStateEnum.READY.value:
        raise InvalidException(message="READY archive jobs are immutable")
    payload = input.model_dump(exclude_unset=True)
    if payload.get("state") == TosArchiveStateEnum.READY.value:
        raise InvalidException(message="Use POST /archives/{id}/complete to finish")
    try:
        await job.update(session, payload)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update archive job: {e}"
        ) from e
    model_file = await ModelFile.one_by_id(session, job.model_file_id)
    if model_file:
        _sync_model_file_archive(model_file, job)
        if model_file.state != ModelFileStateEnum.READY:
            model_file.state = ModelFileStateEnum.READY
        await model_file.update(session)
    return job


@router.post("/archives/{id}/complete", response_model=TosArchiveJobPublic)
async def complete_archive(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: TosArchiveCompleteRequest,
):
    job = await TosArchiveJob.one_by_id(session, id)
    assert_resource_visible(ctx, job, not_found_message=f"Archive job {id} not found")
    _require_assigned_worker(ctx, job)
    if job.state == TosArchiveStateEnum.READY.value:
        return job
    job.state = TosArchiveStateEnum.READY.value
    job.progress = 100
    job.size = input.size
    job.uploaded_bytes = input.uploaded_bytes
    job.error_code = None
    job.state_message = None
    job.provenance = {
        **(job.provenance or {}),
        **input.provenance.model_dump(exclude_none=True),
    }
    await job.update(session)
    model_file = await ModelFile.one_by_id(session, job.model_file_id)
    if model_file:
        _sync_model_file_archive(model_file, job)
        if model_file.state != ModelFileStateEnum.READY:
            model_file.state = ModelFileStateEnum.READY
        await model_file.update(session)
    return job


@router.post("/archives/{id}/retry", response_model=TosArchiveJobPublic)
async def retry_archive(session: SessionDep, ctx: TenantContextDep, id: int):
    job = await TosArchiveJob.one_by_id(session, id)
    assert_resource_visible(ctx, job, not_found_message=f"Archive job {id} not found")
    if job.state == TosArchiveStateEnum.READY.value:
        raise InvalidException(message="READY archive jobs cannot be retried")
    if job.state not in (
        TosArchiveStateEnum.ERROR.value,
        TosArchiveStateEnum.QUEUED.value,
        TosArchiveStateEnum.UPLOADING.value,
        TosArchiveStateEnum.VERIFYING.value,
    ):
        raise InvalidException(message=f"Cannot retry archive in state {job.state}")
    job.state = TosArchiveStateEnum.QUEUED.value
    job.progress = 0
    job.uploaded_bytes = 0
    job.error_code = None
    job.state_message = None
    job.attempt = (job.attempt or 1) + 1
    await job.update(session)
    model_file = await ModelFile.one_by_id(session, job.model_file_id)
    if model_file:
        _sync_model_file_archive(model_file, job)
        await model_file.update(session)
    return job


@router.get("/archives/{id}/transport", response_model=TosTransport)
async def get_archive_transport(session: SessionDep, ctx: TenantContextDep, id: int):
    job = await TosArchiveJob.one_by_id(session, id)
    assert_resource_visible(ctx, job, not_found_message=f"Archive job {id} not found")
    _require_assigned_worker(ctx, job)
    if job.state == TosArchiveStateEnum.READY.value:
        raise InvalidException(message="READY jobs no longer expose transport secrets")
    connection = await TosConnection.one_by_id(session, job.connection_id)
    if not connection:
        raise NotFoundException(message="TOS connection not found")
    return TosTransport(
        endpoint=connection.endpoint,
        region=connection.region,
        bucket=connection.bucket,
        prefix=normalize_prefix(connection.prefix),
        tos_model=job.tos_model,
        tos_version=job.tos_version,
        credential_mode=connection.credential_mode,
        credential_ref=connection.credential_ref,
        access_key=connection.access_key,
        secret_key=connection.secret_key,
        security_token=connection.security_token,
    )


@router.get("/releases", response_model=list[TosReleaseListItem])
async def list_release_catalog(
    session: SessionDep,
    ctx: TenantContextDep,
    connection_id: int = Query(...),
    tos_model: Optional[str] = None,
):
    connection = await load_visible_connection(session, ctx, connection_id)
    prefix = normalize_prefix(connection.prefix)
    try:
        client = tos_storage.SdkTosClient(
            connection.endpoint,
            connection.region,
            access_key=connection.access_key,
            secret_key=connection.secret_key,
            security_token=connection.security_token,
        )
        listed = tos_storage.list_releases(client, connection.bucket, prefix)
    except tos_storage.TosStorageError as exc:
        raise InvalidException(
            message=str(exc),
            details={"error_code": classify_archive_error(exc)},
        ) from exc
    items = []
    for entry in listed:
        if tos_model and entry["tos_model"] != tos_model:
            continue
        items.append(
            TosReleaseListItem(
                connection_id=connection.id,
                tos_model=entry["tos_model"],
                tos_version=entry["tos_version"],
                artifact_uri=artifact_uri(
                    connection.bucket,
                    prefix,
                    entry["tos_model"],
                    entry["tos_version"],
                ),
            )
        )
    return items


@router.get("/releases/detail", response_model=TosReleasePublic)
async def get_release_detail(
    session: SessionDep,
    ctx: TenantContextDep,
    connection_id: int = Query(...),
    tos_model: str = Query(...),
    tos_version: str = Query(...),
):
    connection = await load_visible_connection(session, ctx, connection_id)
    prefix = normalize_prefix(connection.prefix)
    try:
        location = tos_storage.TosLocation(
            endpoint=connection.endpoint,
            region=connection.region,
            bucket=connection.bucket,
            model=tos_model,
            version=tos_version,
            release_path=tos_storage.resolve_release_path(
                tos_model, tos_version, prefix=prefix
            ),
        )
        if not object_key_in_prefix(prefix, location.manifest_key):
            raise InvalidException(
                message="release is outside the connection prefix",
                details={"error_code": TosArchiveErrorCode.PREFIX_DENIED.value},
            )
        client = tos_storage.SdkTosClient(
            location.endpoint,
            location.region,
            access_key=connection.access_key,
            secret_key=connection.secret_key,
            security_token=connection.security_token,
        )
        manifest = tos_storage.fetch_manifest(location, client)
    except tos_storage.TosStorageError as exc:
        raise InvalidException(
            message=str(exc),
            details={"error_code": classify_archive_error(exc)},
        ) from exc
    files = [
        TosReleaseFilePublic(
            path=entry["path"], size=entry["size"], sha256=entry["sha256"]
        )
        for entry in manifest["files"]
    ]
    return TosReleasePublic(
        connection_id=connection.id,
        bucket=connection.bucket,
        prefix=prefix,
        tos_model=tos_model,
        tos_version=tos_version,
        artifact_uri=location.uri,
        schema_version=manifest.get("schema_version"),
        files=files,
        total_bytes=tos_storage.manifest_total_size(manifest),
        gguf=looks_gguf([f.path for f in files]),
        backend_hint=manifest.get("backend_hint"),
    )
