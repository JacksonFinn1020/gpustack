from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.streaming import tenant_streaming
from gpustack.api.tenant import (
    assert_org_owned_writable,
    assert_resource_visible,
    tenant_list_conditions,
    validate_owner_principal,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.schemas.tos import (
    TosConnection,
    TosConnectionCreate,
    TosConnectionListParams,
    TosConnectionPublic,
    TosConnectionUpdate,
    TosConnectionsPublic,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.tos_bindings import (
    connection_scope,
    public_access_key,
    seal_connection_secrets,
)
from gpustack.utils.tos_scope import enforce_endpoint_trust, normalize_prefix
from gpustack.utils.tos_secrets import TosSecretError

router = APIRouter()


async def _redact_connection_event(event) -> None:
    data = getattr(event, "data", None)
    if data is None or isinstance(data, dict):
        return
    if hasattr(data, "access_key"):
        data.access_key = public_access_key(getattr(data, "access_key", None))
    for name in ("secret_key", "security_token"):
        if hasattr(data, name):
            setattr(data, name, None)


def _public(row: TosConnection) -> TosConnectionPublic:
    data = row.model_dump()
    data["access_key"] = public_access_key(row.access_key)
    data.pop("secret_key", None)
    data.pop("security_token", None)
    data["scope"] = connection_scope(row.prefix)
    return TosConnectionPublic.model_validate(data)


@router.get("/connections", response_model=TosConnectionsPublic)
async def list_connections(
    ctx: TenantContextDep,
    params: TosConnectionListParams = Depends(),
):
    fields = {"deleted_at": None}
    if params.watch:
        return StreamingResponse(
            tenant_streaming(
                TosConnection,
                ctx,
                fields=fields,
                event_transform=_redact_connection_event,
            ),
            media_type="text/event-stream",
        )
    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, TosConnection)
        page = await TosConnection.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )
        page.items = [_public(item) for item in page.items]
        return page


@router.get("/connections/{id}", response_model=TosConnectionPublic)
async def get_connection(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await TosConnection.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"TOS connection {id} not found")
    assert_resource_visible(
        ctx, existing, not_found_message=f"TOS connection {id} not found"
    )
    return _public(existing)


@router.post("/connections", response_model=TosConnectionPublic)
async def create_connection(
    session: SessionDep, ctx: TenantContextDep, input: TosConnectionCreate
):
    if input.owner_principal_id is None:
        input.owner_principal_id = ctx.current_principal_id or platform_principal_id()
    validate_owner_principal(
        input.owner_principal_id, ctx, resource_label="TOS connection"
    )
    input.endpoint = enforce_endpoint_trust(
        input.endpoint, credential_mode=input.credential_mode
    )
    input.prefix = normalize_prefix(input.prefix)
    try:
        if input.credential_mode == "env":
            input.access_key = None
            input.secret_key = None
            input.security_token = None
            input.credential_ref = input.credential_ref or "env"
        else:
            sealed = seal_connection_secrets(
                {
                    "access_key": input.access_key,
                    "secret_key": input.secret_key,
                    "security_token": input.security_token,
                }
            )
            input.access_key = sealed["access_key"]
            input.secret_key = sealed["secret_key"]
            input.security_token = sealed["security_token"]
    except TosSecretError as exc:
        raise InvalidException(message=str(exc)) from exc
    existing = await TosConnection.one_by_fields(
        session,
        {
            "deleted_at": None,
            "name": input.name,
            "owner_principal_id": input.owner_principal_id,
        },
    )
    if existing:
        raise AlreadyExistsException(
            message=f"TOS connection with name '{input.name}' already exists."
        )
    try:
        created = await TosConnection.create(
            session, input.model_dump(exclude={"allow_bucket_root"})
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create TOS connection: {e}"
        ) from e
    return _public(created)


@router.put("/connections/{id}", response_model=TosConnectionPublic)
async def update_connection(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: TosConnectionUpdate,
):
    existing = await TosConnection.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"TOS connection {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="TOS connection")
    payload = input.model_dump(exclude_unset=True)
    allow_bucket_root = payload.pop("allow_bucket_root", None)
    if "endpoint" in payload and payload["endpoint"]:
        mode = payload.get("credential_mode") or existing.credential_mode
        payload["endpoint"] = enforce_endpoint_trust(
            payload["endpoint"], credential_mode=mode
        )
    if "prefix" in payload:
        payload["prefix"] = normalize_prefix(payload["prefix"])
        if payload["prefix"] == "" and not allow_bucket_root:
            raise InvalidException(
                message="empty prefix grants the whole bucket; set allow_bucket_root=true"
            )
    try:
        if payload.get("credential_mode") == "env":
            payload["access_key"] = None
            payload["secret_key"] = None
            payload["security_token"] = None
            payload["credential_ref"] = payload.get("credential_ref") or "env"
        else:
            payload = seal_connection_secrets(payload)
            if "secret_key" in payload and payload["secret_key"] is None:
                payload.pop("secret_key")
    except TosSecretError as exc:
        raise InvalidException(message=str(exc)) from exc
    try:
        await existing.update(session, payload)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update TOS connection: {e}"
        ) from e
    return _public(existing)


@router.delete("/connections/{id}")
async def delete_connection(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await TosConnection.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"TOS connection {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="TOS connection")
    try:
        await existing.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete TOS connection: {e}"
        ) from e
