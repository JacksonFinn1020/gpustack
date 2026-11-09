# TOS bidirectional API contract

Machine-readable source of truth: `GET /v2/openapi.json` (enable
`GPUSTACK_EXTENDED_API_DOCS` if your docs surface hides extra routes). This
page is the operator-facing typed contract for the official UI and other
clients. Do not invent parallel field names.

**Auth:** Org Owner (`require_org_role(OWNER)`) for connections, catalog,
archive create/retry, and server ingest. Assigned **worker** system principals
may list/get their jobs, PUT progress, GET transport secrets, and POST
complete. Transport is never returned to Org users.

**Compatibility:** Existing `source=huggingface|model_scope|local_path|tos`
records keep working. `tos_endpoint` / `tos_region` / `tos_bucket` /
`tos_model` / `tos_version` / `tos_release_path` remain valid. New optional
`tos_connection_id` fills those coordinates from a stored connection. Hugging
Face / ModelScope / Local Path download and deploy paths are unchanged.

## Identities (keep these four separate)

| Identity | Where it lives |
| --- | --- |
| Original source | `provenance.origin_*` (HF repo, ModelScope id, or local path) |
| TOS artifact | `tos://{bucket}/{prefix}/{model}/{version}` plus manifest SHA |
| This deployment source | Model / ModelFile `source` + TOS or Hub fields |
| Local worker cache | `resolved_paths` on the READY ModelFile |

Deleting a local ModelFile or a deployment does **not** delete the remote TOS
release.

## Connections

`POST /v2/tos/connections`

```json
{
  "name": "prod-weights",
  "endpoint": "https://tos-cn-beijing.volces.com",
  "region": "cn-beijing",
  "bucket": "gpustack-weights",
  "prefix": "org-a/models",
  "credential_mode": "inline",
  "access_key": "AKLT...",
  "secret_key": "..."
}
```

`credential_mode=env` with `credential_ref=env` uses `TOS_ACCESS_KEY` /
`TOS_SECRET_KEY` on the **process that performs the TOS call**: the assigned
worker for archive/pull, the control plane for catalog/size/complete verify.
Each of those processes must set `TOS_ENDPOINT` (or `TOS_ALLOWED_ENDPOINTS`)
to the **same** https host as the connection. Worker-only env is enough for
pull/archive; catalog and deploy-time manifest checks on the control plane
need the same pin **on the control plane**, or use `credential_mode=inline`.
Inline keys never fall back to process env. Legacy `tos_endpoint` without
`tos_connection_id` is env-only and must match the pin list.

Inline `access_key` / `secret_key` / `security_token` are sealed at rest
(`gsk1:` Fernet from **`GPUSTACK_TOS_SECRET_KEY` only** — not the JWT secret)
and decrypted only on assigned-worker transport / control-plane catalog.
Plaintext is never stored. Rotate with `GPUSTACK_TOS_SECRET_KEY_PREVIOUS`.
Empty `prefix` is whole-bucket scope and requires `allow_bucket_root=true`;
`GET` connections include `scope=prefix|bucket`.

Read (`GET /v2/tos/connections`, `GET /v2/tos/connections/{id}`) returns
redacted `access_key` and **never** `secret_key` or `security_token`. Catalog
listing is limited to that connection's bucket+prefix.

## Archive (worker cache → TOS)

`POST /v2/tos/archives` (Owner)

```json
{
  "model_file_id": 42,
  "connection_id": 1,
  "tos_model": "meta/llama-3-8b",
  "tos_version": "v1",
  "paths": null
}
```

Single-file GGUF: `"paths": ["Llama-3-8B-Q4_K_M.gguf"]`. Version is explicit
and immutable. Same `(connection_id, tos_model, tos_version)` returns `409`
`AlreadyExists`. If the existing job is `error`, `details.error_code` is
`version_retry_required` with `details.retry_job_id` — call
`POST /v2/tos/archives/{id}/retry`, do not create a new job. In-progress jobs
use `concurrent_version`. READY jobs use `immutable_conflict`.

States: `queued` → `uploading` → `verifying` → `ready` | `error`.

| Method | Path | Who |
| --- | --- | --- |
| GET | `/v2/tos/archives?worker_id=&model_file_id=&state=` | Owner, assigned worker (SYSTEM callers are forced to their own `worker_id`) |
| GET | `/v2/tos/archives/{id}` | Owner, assigned worker |
| PUT | `/v2/tos/archives/{id}` | Assigned worker (progress; cannot set `ready`) |
| POST | `/v2/tos/archives/{id}/complete` | Assigned worker (atomic finish) |
| POST | `/v2/tos/archives/{id}/retry` | Owner (does not delete READY local files) |
| GET | `/v2/tos/archives/{id}/transport` | Assigned worker only |
| GET | `/v2/tos/pull/{model_file_id}/transport` | Assigned worker only (import/deploy) |

`error_code` values: `incomplete_upload`, `immutable_conflict`,
`concurrent_version`, `version_retry_required`, `checksum_mismatch`,
`disk_full`, `worker_unreachable`, `invalid_credentials`, `path_escape`,
`prefix_denied`, `not_ready`, `source_mismatch`, `format_unsupported`,
`connection_not_found`, `manifest_mismatch`.

Objects are uploaded first; `manifest.json` is written last with TOS
`forbid_overwrite`. A crash mid-upload leaves no catalog entry. Retry **re-hashes**
existing content-addressed objects before skipping them; size/SHA mismatch
re-uploads. `POST .../complete` is idempotent when the job is already `ready`,
requires `provenance.manifest_sha256`, **re-fetches the committed remote
manifest**, and keeps `origin_*` from the ModelFile record (worker payloads
and `PUT /archives/{id}` cannot replace them). Workers reclaim `queued`,
`uploading`, and `verifying` jobs after restart. SDK `forbid_overwrite` and
cross-process DB commit vs object PUT atomicity are **not** proven on live TOS.

## Catalog and deploy

`GET /v2/tos/releases?connection_id=1&tos_model=optional`

`GET /v2/tos/releases/detail?connection_id=1&tos_model=qwen3&tos_version=v1`

Then reuse:

- `POST /v2/model-files` with `source=tos`, `tos_connection_id` (or legacy
  coordinates), `worker_id`
- `POST /v2/models` with `source=tos`, same coordinates, existing
  `backend`, `cluster_id`, `gpu_selector`, `replicas`

`GET /v2/tos/releases` fills `gguf`, `object_count`, `total_bytes` from the
trusted manifest and sets `catalog_verified=true`. If the manifest cannot be
read, those fields are null and `catalog_verified=false` (unknown — not a
verified hit). `POST /v2/models` stamps provenance from that manifest and
**ignores client `provenance.gguf`**. TOS GGUF cannot use vLLM / SGLang /
MindIE / VoxBox. `trust_remote_code` is never enabled by this path. There is
**no** `/tos/browse` route; catalog is `GET /v2/tos/releases` and
`GET /v2/tos/releases/detail`.

## Auto-archive (HF / ModelScope)

On `POST /v2/models` or `POST /v2/model-files`, optional:

```json
{
  "source": "huggingface",
  "huggingface_repo_id": "Qwen/Qwen3-8B",
  "auto_archive": {
    "enabled": true,
    "connection_id": 1,
    "tos_model": "qwen3-8b",
    "tos_version": "hub-2026-11-09",
    "paths": null
  }
}
```

Default is off. Archive state is `archive_state` on the ModelFile and is
independent of download `state`. Archive failure must not set ModelFile
`state=error` after a successful READY download. Retry with
`POST /v2/tos/archives/{id}/retry`.

## Resume and cost

TOS **download** skips files whose local size and SHA256 already match the
manifest (including after worker restart, using `.deployment-manifest.json`).
Partial `.tos-tmp/*.part` files are discarded and the object is fetched again.
There is no multipart byte-range resume in this release; for multi-tens-of-GB
weights that is a full object retry. **Upload** re-hashes existing content-addressed keys and only skips on SHA+size match.

S3-compatible APIs are **not** claimed unsupported; this implementation uses
the native TOS SDK because `forbid_overwrite` is available and tested against
`FakeTos`. Equivalence on S3 has not been proven in this repo.

## UI consumption checklist

The UI (separate repository; do not change it in this PR) can bind:

- connection picker → `GET /v2/tos/connections`
- version picker → `GET /v2/tos/releases?connection_id=`
- target worker → existing model-file / deploy worker fields
- manual archive → `POST /v2/tos/archives` (directory or `paths`)
- auto archive → `auto_archive` on create, list `archive_state`
- deploy → existing `backend`, `cluster_id`, `gpu_selector` plus `source=tos`
