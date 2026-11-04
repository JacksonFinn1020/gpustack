# gpustack-release bridge schema

Phase 1 converter output. One JSON object maps a model-deploy catalog/deploy
pair onto a GPUStack `ModelCreate` body (`source=local_path`).

See the machine-readable copy in
[`schema/gpustack-release.schema.json`](schema/gpustack-release.schema.json)
and the operator guide in
[`docs/integrations/integrate-with-model-deploy.md`](../../docs/integrations/integrate-with-model-deploy.md).

## Top-level object

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | int | yes | Currently `1`. |
| `source_kind` | string | yes | Always `model-deploy`. |
| `model_deploy` | object | yes | Provenance: catalog key, weight version, runtime. |
| `model_create` | object | yes | GPUStack write body. No server-managed ids. |
| `probe` | object | no | model-deploy health JSON; **not** a `ModelCreate` field. |
| `notes` | string[] | no | Operator reminders (weight sync, GPU mapping). |

## `model_deploy`

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | Catalog key / model-deploy `--model`. |
| `version` | string | yes | Immutable weight version; directory name under `model_root`. |
| `runtime` | string | yes | `vllm` or `sglang` in phase 1. |
| `served_model_name` | string | no | Catalog / `service.json` served name. |
| `model_root` | string | no | Default `/data/models`. |
| `physical_gpus` | string | no | Host indices from `runtime.gpus` (for example `"0,1"`). |

## `model_create` (GPUStack)

Subset of `ModelCreate` used by this adapter. Extra GPUStack fields may be
added by hand before apply.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | Deployment name (`served_model_name` or catalog key). |
| `source` | string | yes | Always `local_path`. |
| `local_path` | string | yes | `{model_root}/{model}/{version}` on every target worker. |
| `backend` | string | yes | `vLLM` or `SGLang`. |
| `backend_parameters` | string[] | no | `--tensor-parallel-size` / `--tp` and `extra_args`. |
| `env` | object | no | Merged catalog + deploy `runtime.environment`. Offline flags kept. |
| `replicas` | int | no | Default `1`. |
| `gpu_selector` | object | no | `{ "gpu_ids": ["worker:cuda:0"] }` when `--worker` is set. |
| `categories` | string[] | no | Default `["llm"]`. |
| `enable_model_route` | bool | no | Default `true`. |
| `description` | string | no | Short provenance string. |
| `cluster_name` | string | no | YAML import only; not sent on `POST /v2/models` (this tree's management prefix; not OpenAI `GET /v1/models`). |

Omitted on purpose (server-managed or environment-bound): `id`, `cluster_id`,
`owner_principal_id`, `access_policy`, `ready_replicas`, `revision_history_limit`.

## `probe`

| Field | Type | Notes |
|---|---|---|
| `health_path` | string | For example `/v1/models`. |
| `health_expect_json` | object | model-deploy JSON equality (not GPUStack instance state). |
| `health_require_nonempty_data` | bool | When true, `data` must be a non-empty list. |

GPUStack `Running` is not this probe. After apply, operators should still hit
the OpenAI-compatible `/v1/models` (or the catalog path) and compare JSON.

## Unsupported in phase 1

`custom-http`, `python-adapter`, and any other catalog runtime fail conversion
with a pointer to this schema and
[Using Custom Inference Backends](../../docs/tutorials/using-custom-backends.md).
Do not silently map those launchers to built-in vLLM or SGLang.
