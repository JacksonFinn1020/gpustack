from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
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
from gpustack.server.tos_bindings import redact_access_key
from gpustack.utils.tos_scope import https_endpoint, normalize_prefix

router = APIRouter()


def _public(row: TosConnection) -> TosConnectionPublic:
    data = row.model_dump()
    data["access_key"] = redact_access_key(row.access_key)
    data.pop("secret_key", None)
    data.pop("security_token", None)
    return TosConnectionPublic.model_validate(data)


@router.get("/connections", response_model=TosConnectionsPublic)
async def list_connections(
    ctx: TenantContextDep,
    params: TosConnectionListParams = Depends(),
):
    fields = {"deleted_at": None}
    if params.watch:
        return StreamingResponse(
            tenant_streaming(TosConnection, ctx, fields=fields),
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
    input.endpoint = https_endpoint(input.endpoint)
    input.prefix = normalize_prefix(input.prefix)
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
        created = await TosConnection.create(session, input)
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
    if "endpoint" in payload and payload["endpoint"]:
        payload["endpoint"] = https_endpoint(payload["endpoint"])
    if "prefix" in payload:
        payload["prefix"] = normalize_prefix(payload["prefix"])
    if "secret_key" in payload and payload["secret_key"] is None:
        payload.pop("secret_key")
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
