from enum import Enum
from typing import ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator
from sqlalchemy import Column, String, Text, UniqueConstraint
from sqlmodel import Field as SQLField, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, ListParams, PaginatedList, PublicFields


class TosCredentialMode(str, Enum):
    INLINE = "inline"
    ENV = "env"

    def __str__(self):
        return self.value


class TosArchiveStateEnum(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    VERIFYING = "verifying"
    READY = "ready"
    ERROR = "error"

    def __str__(self):
        return self.value


class TosArchiveErrorCode(str, Enum):
    INCOMPLETE_UPLOAD = "incomplete_upload"
    IMMUTABLE_CONFLICT = "immutable_conflict"
    CONCURRENT_VERSION = "concurrent_version"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    DISK_FULL = "disk_full"
    WORKER_UNREACHABLE = "worker_unreachable"
    INVALID_CREDENTIALS = "invalid_credentials"
    PATH_ESCAPE = "path_escape"
    PREFIX_DENIED = "prefix_denied"
    NOT_READY = "not_ready"
    SOURCE_MISMATCH = "source_mismatch"
    FORMAT_UNSUPPORTED = "format_unsupported"
    CONNECTION_NOT_FOUND = "connection_not_found"
    VERSION_RETRY_REQUIRED = "version_retry_required"
    MANIFEST_MISMATCH = "manifest_mismatch"

    def __str__(self):
        return self.value


class TosProvenance(SQLModel):
    """Four-way identity: origin source, TOS artifact, this deploy, local cache."""

    origin_source: Optional[str] = None
    origin_huggingface_repo_id: Optional[str] = None
    origin_huggingface_filename: Optional[str] = None
    origin_model_scope_model_id: Optional[str] = None
    origin_model_scope_file_path: Optional[str] = None
    origin_local_path: Optional[str] = None
    origin_worker_id: Optional[int] = None
    origin_model_file_id: Optional[int] = None
    upstream_revision: Optional[str] = None
    selected_paths: Optional[List[str]] = None
    manifest_sha256: Optional[str] = None
    manifest_bytes: Optional[int] = None
    object_count: Optional[int] = None
    artifact_uri: Optional[str] = None
    gguf: Optional[bool] = None
    backend_hint: Optional[str] = None


class AutoArchiveSpec(SQLModel):
    enabled: bool = False
    connection_id: Optional[int] = None
    tos_model: Optional[str] = None
    tos_version: Optional[str] = None
    paths: Optional[List[str]] = None


class TosConnectionBase(SQLModel):
    name: str = Field(min_length=1, max_length=255)
    endpoint: str = Field(
        description=(
            "HTTPS Volcengine-compatible TOS endpoint. "
            "Credentials must not be embedded."
        )
    )
    region: str
    bucket: str
    prefix: str = Field(
        default="",
        description=(
            "Relative object prefix this Org may read and write. "
            "Empty is the whole bucket and requires allow_bucket_root."
        ),
    )
    credential_mode: str = Field(default=TosCredentialMode.INLINE.value)
    credential_ref: Optional[str] = Field(
        default=None,
        description=(
            "When credential_mode=env, the value 'env' means the assigned "
            "worker uses TOS_* environment variables. Other refs are reserved."
        ),
    )
    access_key: Optional[str] = None
    description: Optional[str] = None
    owner_principal_id: Optional[int] = None


class TosConnectionCreate(TosConnectionBase):
    secret_key: Optional[str] = Field(
        default=None,
        description="Write-only. Never returned on read.",
    )
    security_token: Optional[str] = Field(default=None, description="Write-only STS.")
    allow_bucket_root: bool = Field(
        default=False,
        description=(
            "Required when prefix is empty. Empty prefix authorizes the "
            "entire bucket, not a model subdirectory."
        ),
    )

    @model_validator(mode="after")
    def credentials_present(self):
        mode = self.credential_mode or TosCredentialMode.INLINE.value
        prefix = (self.prefix or "").strip()
        if not prefix and not self.allow_bucket_root:
            raise ValueError(
                "empty prefix grants the whole bucket; set allow_bucket_root=true"
            )
        if mode == TosCredentialMode.ENV.value:
            if (self.credential_ref or "env") != "env":
                raise ValueError("credential_ref for env mode must be 'env'")
            return self
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "inline TOS connections require access_key and secret_key on create"
            )
        return self


class TosConnectionUpdate(SQLModel):
    name: Optional[str] = None
    endpoint: Optional[str] = None
    region: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    allow_bucket_root: Optional[bool] = None
    credential_mode: Optional[str] = None
    credential_ref: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = Field(default=None)
    security_token: Optional[str] = None
    description: Optional[str] = None


class TosConnection(TosConnectionBase, BaseModelMixin, table=True):
    __tablename__ = "tos_connections"
    __table_args__ = (
        UniqueConstraint(
            "owner_principal_id", "name", name="uix_tos_connections_name_per_owner"
        ),
    )
    id: Optional[int] = SQLField(default=None, primary_key=True)
    secret_key: Optional[str] = SQLField(default=None, exclude=True)
    security_token: Optional[str] = SQLField(default=None, exclude=True)


class TosConnectionPublic(TosConnectionBase, PublicFields):
    id: int
    access_key: Optional[str] = Field(
        default=None,
        description="Present when stored inline; secret_key is never included.",
    )
    scope: str = Field(
        default="prefix",
        description="prefix for a jailed prefix; bucket when prefix is empty.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "name": "prod-weights",
                    "endpoint": "https://tos-cn-beijing.volces.com",
                    "region": "cn-beijing",
                    "bucket": "gpustack-weights",
                    "prefix": "org-a/models",
                    "credential_mode": "inline",
                    "credential_ref": None,
                    "access_key": "AKLT***",
                    "description": "Org A model artifacts",
                    "owner_principal_id": 12,
                }
            ]
        }
    )


class TosConnectionListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "created_at",
        "updated_at",
    ]


TosConnectionsPublic = PaginatedList[TosConnectionPublic]


class TosArchiveJobBase(SQLModel):
    model_file_id: int
    connection_id: int
    worker_id: int
    tos_model: str
    tos_version: str
    paths: Optional[List[str]] = SQLField(sa_column=Column(JSON), default=None)
    state: str = SQLField(
        default=TosArchiveStateEnum.QUEUED.value,
        sa_column=Column(String(length=32), nullable=False),
    )
    progress: float = 0
    size: Optional[int] = None
    uploaded_bytes: Optional[int] = None
    state_message: Optional[str] = SQLField(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_code: Optional[str] = SQLField(
        default=None, sa_column=Column(String(length=64), nullable=True)
    )
    attempt: int = 1
    provenance: Optional[Dict] = SQLField(sa_column=Column(JSON), default=None)
    cluster_id: Optional[int] = None
    owner_principal_id: Optional[int] = None


class TosArchiveJobCreate(SQLModel):
    model_file_id: int = Field(description="READY ModelFile on the target worker")
    connection_id: int
    tos_model: str = Field(description="Immutable catalog model id, e.g. org/llama-3")
    tos_version: str = Field(description="Explicit immutable version, e.g. 2026-11-09")
    paths: Optional[List[str]] = Field(
        default=None,
        description=(
            "Relative files under the READY cache. "
            "Null archives the full resolved tree."
        ),
    )
    backend_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional backend name recorded in provenance; "
            "not executed during archive."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_file_id": 42,
                    "connection_id": 1,
                    "tos_model": "meta/llama-3-8b",
                    "tos_version": "v1",
                    "paths": None,
                },
                {
                    "model_file_id": 42,
                    "connection_id": 1,
                    "tos_model": "meta/llama-3-8b",
                    "tos_version": "gguf-q4",
                    "paths": ["Llama-3-8B-Q4_K_M.gguf"],
                },
            ]
        }
    )


class TosArchiveJobUpdate(SQLModel):
    state: Optional[str] = None
    progress: Optional[float] = None
    size: Optional[int] = None
    uploaded_bytes: Optional[int] = None
    state_message: Optional[str] = None
    error_code: Optional[str] = None


class TosArchiveCompleteRequest(SQLModel):
    provenance: TosProvenance
    size: int
    uploaded_bytes: int


class TosArchiveJob(TosArchiveJobBase, BaseModelMixin, table=True):
    __tablename__ = "tos_archive_jobs"
    __table_args__ = (
        UniqueConstraint(
            "connection_id",
            "tos_model",
            "tos_version",
            name="uix_tos_archive_connection_model_version",
        ),
    )
    id: Optional[int] = SQLField(default=None, primary_key=True)
    model_file_id: int = SQLField(foreign_key="model_files.id")
    connection_id: int = SQLField(foreign_key="tos_connections.id")
    worker_id: int = SQLField(foreign_key="workers.id")


class TosArchiveJobPublic(TosArchiveJobBase, PublicFields):
    id: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 9,
                    "model_file_id": 42,
                    "connection_id": 1,
                    "worker_id": 7,
                    "tos_model": "meta/llama-3-8b",
                    "tos_version": "v1",
                    "paths": None,
                    "state": "uploading",
                    "progress": 41.5,
                    "size": 16000000000,
                    "uploaded_bytes": 6640000000,
                    "error_code": None,
                    "attempt": 1,
                }
            ]
        }
    )


class TosArchiveJobListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "created_at",
        "updated_at",
        "state",
        "tos_model",
        "tos_version",
    ]


TosArchiveJobsPublic = PaginatedList[TosArchiveJobPublic]


class TosTransport(SQLModel):
    """Worker-only payload. Never returned to Org users."""

    endpoint: str
    region: str
    bucket: str
    prefix: str
    tos_model: str
    tos_version: str
    credential_mode: str
    credential_ref: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    security_token: Optional[str] = None


class TosReleaseFilePublic(SQLModel):
    path: str
    size: int
    sha256: str


class TosReleasePublic(SQLModel):
    connection_id: int
    bucket: str
    prefix: str
    tos_model: str
    tos_version: str
    artifact_uri: str
    schema_version: Optional[int] = None
    files: List[TosReleaseFilePublic] = []
    total_bytes: int = 0
    gguf: bool = False
    backend_hint: Optional[str] = None


class TosReleaseListItem(SQLModel):
    connection_id: int
    tos_model: str
    tos_version: str
    artifact_uri: str
    object_count: Optional[int] = None
    total_bytes: Optional[int] = None
    gguf: Optional[bool] = None
    catalog_verified: bool = False
