# TOS field acceptance (A–H) and safety premises

This is the operator checklist for bidirectional TOS (archive READY worker
cache to TOS, then import a version onto assigned workers). Cloud CI in this
repository uses **mocks** (`FakeTos`, schema/unit tests). It does **not**
prove live TOS, GPU, or hub-outage behavior.

Code baseline for this work: git SHA of the branch that ships it (record at
deploy time). Historical labels such as `tos-b6ae1ae` / a private v2 fork are
**not** current production.

## Mock vs real

| Item | Mock (this repo) | Real field |
| --- | --- | --- |
| Content-addressed upload, immutable version, subset file | `FakeTos` unit tests | Live bucket |
| SHA download, cache reuse, lock | `FakeTos` + temp dirs | Worker disk |
| Prefix / path escape / redaction | unit tests | Live IAM |
| API Owner vs worker transport | SYSTEM_ROUTES contract tests | Live tokens |
| HF/MS/local_path regression | existing pytest | — |
| GPU launch, multi-replica locality | not executed here | cluster |
| Invalid AK/SK, disk full, worker down | error classification unit tests | inject on site |
| Hub disconnected after TOS import | not executed here | disconnect upstream |

Do not treat a green `pytest` run as A–H PASS on a real cluster.

## A–H coverage

| ID | Intent | This PR | Still field-only |
| --- | --- | --- | --- |
| A | Archive READY directory | FakeTos full-tree upload + ArchiveJob states | Live TOS |
| B | Archive single file (GGUF) | `relatives=[file]` | Live TOS |
| C | Import TOS version → worker READY → schedule | existing TosDownloader + ModelFile | GPU start |
| D | Auto-archive after HF/MS, independent error | controller enqueue + READY preserved | Live hub+TOS |
| E | Concurrent same version | unique constraint + `forbid_overwrite` | two workers |
| F | Retry incomplete upload | objects without manifest then retry | kill mid-PUT |
| G | Auth, redaction, prefix jail | unit + contract tests | stolen token |
| H | Old HF/MS/local records + migration rollback | Alembic downgrade drops new tables/columns | restore DB snapshot |

## Field script (do not run in CI with real credentials)

Set these **on the target worker/server only**, never commit them:

```bash
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_ENDPOINT=https://tos-cn-beijing.volces.com
export TOS_REGION=cn-beijing
export TOS_BUCKET=...
```

Suggested sequence (Owner API + one READY ModelFile):

1. `POST /v2/tos/connections` (inline keys or `credential_mode=env`).
2. Confirm `GET /v2/tos/connections/{id}` has no `secret_key`.
3. `POST /v2/tos/archives` with explicit `tos_version`.
4. Watch `GET /v2/tos/archives/{id}` until `ready`.
5. `GET /v2/tos/releases?connection_id=`.
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
`TOS_*` env on the worker.

## Security premises

- HTTPS endpoints only; credentials never in the URL, logs, or GET bodies for
  Org users.
- Connection `prefix` is a jail for catalog and archive keys.
- Archive scan rejects `..` and escaping symlinks.
- `trust_remote_code` is not flipped on.
- Do not point `endpoint` at an attacker host from an untrusted Org if
  workers would then send stored keys there — treat connections as trusted
  Org-owner config.
