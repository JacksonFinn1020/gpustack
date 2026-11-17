"""Resolve TOS credentials by connection scope.

Process environment keys are used only when ``credential_mode=env`` and the
endpoint is pinned. Inline connections never fall back to ``TOS_ACCESS_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from gpustack.api.exceptions import ForbiddenException, InvalidException
from gpustack.schemas.tos import TosArchiveErrorCode, TosCredentialMode, TosTransport
from gpustack.utils.tos_scope import env_endpoint_allows, https_endpoint
from gpustack.utils.tos_secrets import open_secret
from gpustack.worker.tos_storage import SdkTosClient, TosStorageError


@dataclass(frozen=True)
class TosAccess:
    endpoint: str
    region: str
    bucket: str
    prefix: str
    credential_mode: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    security_token: Optional[str] = None
    credential_ref: Optional[str] = None
    tos_model: Optional[str] = None
    tos_version: Optional[str] = None


def assert_connection_owned_by(
    connection: Any, owner_principal_id: Optional[int]
) -> None:
    """Deny when either owner is missing or the Org ids differ."""
    if (
        connection is None
        or getattr(connection, "deleted_at", None) is not None
        or owner_principal_id is None
        or getattr(connection, "owner_principal_id", None) is None
        or connection.owner_principal_id != owner_principal_id
    ):
        raise ForbiddenException(
            message="TOS connection is not owned by this organization",
            details={"error_code": TosArchiveErrorCode.CONNECTION_NOT_FOUND.value},
        )


def access_from_connection(
    connection: Any,
    *,
    tos_model: Optional[str] = None,
    tos_version: Optional[str] = None,
) -> TosAccess:
    endpoint = https_endpoint(connection.endpoint)
    mode = connection.credential_mode or TosCredentialMode.INLINE.value
    if mode == TosCredentialMode.ENV.value:
        if not env_endpoint_allows(endpoint):
            raise InvalidException(
                message="env credentials cannot be used unless TOS_ENDPOINT "
                "(or TOS_ALLOWED_ENDPOINTS) matches this connection endpoint",
                details={"error_code": TosArchiveErrorCode.INVALID_CREDENTIALS.value},
            )
        return TosAccess(
            endpoint=endpoint,
            region=connection.region,
            bucket=connection.bucket,
            prefix=connection.prefix or "",
            credential_mode=TosCredentialMode.ENV.value,
            credential_ref=connection.credential_ref or "env",
            tos_model=tos_model,
            tos_version=tos_version,
        )
    return TosAccess(
        endpoint=endpoint,
        region=connection.region,
        bucket=connection.bucket,
        prefix=connection.prefix or "",
        credential_mode=TosCredentialMode.INLINE.value,
        access_key=open_secret(connection.access_key),
        secret_key=open_secret(connection.secret_key),
        security_token=open_secret(connection.security_token),
        credential_ref=connection.credential_ref,
        tos_model=tos_model,
        tos_version=tos_version,
    )


def access_from_legacy_source(model: Any) -> TosAccess:
    """Coordinates without ``tos_connection_id`` may only use pinned env keys."""
    endpoint = getattr(model, "tos_endpoint", None)
    if not endpoint:
        raise TosStorageError(
            "TOS source is missing tos_connection_id and tos_endpoint"
        )
    endpoint = https_endpoint(endpoint)
    if not env_endpoint_allows(endpoint):
        raise TosStorageError(
            "legacy TOS coordinates without tos_connection_id may only target "
            "a pinned TOS_ENDPOINT / TOS_ALLOWED_ENDPOINTS host"
        )
    return TosAccess(
        endpoint=endpoint,
        region=getattr(model, "tos_region", None) or "",
        bucket=getattr(model, "tos_bucket", None) or "",
        prefix="",
        credential_mode=TosCredentialMode.ENV.value,
        credential_ref="env",
        tos_model=getattr(model, "tos_model", None),
        tos_version=getattr(model, "tos_version", None),
    )


def access_from_transport(transport: TosTransport) -> TosAccess:
    return TosAccess(
        endpoint=https_endpoint(transport.endpoint),
        region=transport.region,
        bucket=transport.bucket,
        prefix=transport.prefix or "",
        credential_mode=transport.credential_mode,
        access_key=transport.access_key,
        secret_key=transport.secret_key,
        security_token=transport.security_token,
        credential_ref=transport.credential_ref,
        tos_model=transport.tos_model,
        tos_version=transport.tos_version,
    )


def client_from_access(access: TosAccess) -> SdkTosClient:
    if access.credential_mode == TosCredentialMode.ENV.value:
        if not env_endpoint_allows(access.endpoint):
            raise TosStorageError(
                "env credentials cannot be used unless TOS_ENDPOINT "
                "matches this endpoint"
            )
        return SdkTosClient(access.endpoint, access.region, use_env=True)
    if not access.access_key or not access.secret_key:
        raise TosStorageError(
            "inline TOS credentials are required; refusing process environment"
        )
    return SdkTosClient(
        access.endpoint,
        access.region,
        access_key=access.access_key,
        secret_key=access.secret_key,
        security_token=access.security_token,
        use_env=False,
    )


def client_from_transport(transport: TosTransport) -> SdkTosClient:
    return client_from_access(access_from_transport(transport))


def client_for_source(model: Any, client: Optional[Any] = None) -> Any:
    """Worker/CP download client. Never constructs SdkTosClient from env by default."""
    if client is not None:
        return client
    attached = getattr(model, "_tos_access", None)
    if attached is not None:
        return client_from_access(attached)
    if getattr(model, "tos_connection_id", None):
        raise TosStorageError(
            "TOS connection download requires pull transport credentials; "
            "refusing process environment fallback"
        )
    return client_from_access(access_from_legacy_source(model))
