# TOS field acceptance (A–H) and safety premises

This is the operator checklist for bidirectional TOS (archive READY worker
cache to TOS, then import a version onto assigned workers). Cloud CI in this
repository uses **mocks** (`FakeTos`, schema/unit tests). It does **not**
prove live TOS, GPU, or hub-outage behavior.

## Measured field baseline (not this PR)

Read-only observation on the user machine, **2026-11-09 14:40–14:44
Asia/Shanghai**. This is the currently running stack, **not** PR #3.

| Surface | Observed |
| --- | --- |
| Control plane image | `gpustack/gpustack:tos-b6ae1ae` |
| `/version` and package `__git_commit__` | `b6ae1ae` |
| Worker `10150` `/health` and `/ready` | HTTP 200, same `b6ae1ae` |
| Official UI `data-version` | `03837ee` (cannot be mapped to unmerged UI PR head `db6f2fe1`) |
| Live OpenAPI | `POST /v2/tos/releases`, `GET /v2/tos/sources` only |
| Live OpenAPI missing | `archives`, `connections`, catalog `releases` / `releases/detail` (no browse) |

**PR #3 / `ace966e1` and later commits on `cursor/tos-bidirectional-archive-9861` are not deployed.** Do not treat `tos-b6ae1ae` as this bidirectional archive implementation. Historical fork labels remain historical.

Code to ship after this review: record the merge-candidate SHA of
`cursor/tos-bidirectional-archive-9861` at deploy time.

## Mock vs real

| Item | Mock (this repo) | Real field |
| --- | --- | --- |
| Content-addressed upload, SHA before skip, subset file | `FakeTos` | Live bucket + SDK |
| SHA download, cache reuse, lock | `FakeTos` + temp dirs | Worker disk |
| Prefix / path escape / env endpoint pin / redaction | unit tests | Live IAM + DNS |
| Assigned-worker transport vs Org user | handler unit tests | Live tokens |
| HF/MS/local_path regression | existing pytest | — |
| GPU launch, multi-replica locality | not executed here | cluster |
| Invalid AK/SK, disk full, worker down | error classification + reclaim states | inject on site |
| Hub disconnected after TOS import | not executed here | disconnect upstream |
| Native TOS `forbid_overwrite`, HTTP redirects, DB+PUT atomicity | FakeTos / code review | **pending live SDK** |

Do not treat a green `pytest` run as A–H PASS on a real cluster. Parametrized
`test_system_route_access` counts (hundreds of method/path/kind rows) only
prove **management allowlisting**, not archive upload or auto_archive.

## A–H coverage (user matrix)

| ID | Intent | This PR (mock) | Still field-only |
| --- | --- | --- | --- |
| A | HF manual archive | FakeTos upload from READY HF cache | Live TOS + hub |
| B | MS auto-archive | controller enqueue + owner check | Live hub+TOS |
| C | TOS import | worker pull transport + SHA download | Live TOS |
| D | TOS deploy | POST models stamps manifest; backend reject | GPU start |
| E | Upstream hub down, TOS deploy still works | not executed here | disconnect hub |
| F | Cache reuse | `.deployment-manifest.json` SHA skip | worker disk |
| G | Six-fault recovery (upload kill, download kill, bad keys, disk full, checksum, worker down) | error classify + retry states | inject on site |
| H | Old HF/MS/local_path + legacy TOS coordinates | alembic keeps old rows; sources still valid | restore DB snapshot |

## Field script (do not run in CI with real credentials)

Set these **on the target worker/server only**, never commit them:

```bash
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_ENDPOINT=https://tos-cn-beijing.volces.com
export TOS_REGION=cn-beijing
export TOS_BUCKET=...
```

Suggested sequence **after this code is deployed** (it is not on `tos-b6ae1ae`):

1. `POST /v2/tos/connections` (inline keys or `credential_mode=env`).
2. Confirm `GET /v2/tos/connections/{id}` has no `secret_key`.
3. `POST /v2/tos/archives` with explicit `tos_version`.
4. Watch `GET /v2/tos/archives/{id}` until `ready`.
5. `GET /v2/tos/releases?connection_id=` (not a browse path).
6. `POST /v2/models` `source=tos` + `tos_connection_id` + `gpu_selector`.
7. Confirm ModelFile `resolved_paths` on the assigned worker; start inference.
8. Optional: disable hub network and send a completion request.

Faults to inject **on site only** (not from this agent): upload kill,
download kill, bad keys, fill disk, checksum tamper, stop worker. Retry must
leave the original READY cache in place.

## Migration / rollback

Revision `e1f2a3b4c5d6` (after `d9f1a2b3c4e5`) adds `tos_connections`,
`tos_archive_jobs`, and provenance columns. `alembic downgrade e1f2a3b4c5d6`
drops those tables/columns. It does not delete objects already in TOS.
Old models without `tos_connection_id` keep using per-row coordinates and
`TOS_*` env on the worker. Default `cleanup_on_delete` only removes **local**
cache; it does not call TOS delete.

## Security premises

- HTTPS endpoints only; private/loopback/link-local IP literals and
  `localhost` / metadata hostnames are rejected. SDK HTTP redirects are
  **unverified live**.
- `credential_mode=env` will not mint transport unless `TOS_ENDPOINT` matches.
- Connection secrets are Fernet-sealed at rest with `GPUSTACK_TOS_SECRET_KEY`
  (fail closed; JWT is not used). They are decrypted only on assigned-worker
  transport and control-plane catalog. Watch/list redact `access_key` and
  never emit `secret_key`. Unsealed rows cannot be opened; recreate the
  connection. Rotate with `GPUSTACK_TOS_SECRET_KEY_PREVIOUS`.
- Endpoint trust: HTTPS, no credentials/path, block private IP literals and
  metadata hosts. `credential_mode=env` and
  `GPUSTACK_TOS_REQUIRE_ENDPOINT_ALLOWLIST=true` require `TOS_ENDPOINT` /
  `TOS_ALLOWED_ENDPOINTS`. DNS rebinding and SDK redirects are **unproven**
  on live TOS; the executable gate is the allowlist.
- Connection `prefix` jails catalog keys and archive object keys.
- Archive scan rejects `..` and escaping symlinks.
- `trust_remote_code` is not flipped on.
- Do not point `endpoint` at an attacker host from an untrusted Org if
  workers would then send **inline** stored keys there — treat connections as
  trusted Org-owner config. Env keys are additionally pinned.
