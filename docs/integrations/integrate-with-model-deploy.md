# Integrate with model-deploy

[model-deploy](https://gitee.com) is a release toolkit that uploads immutable
weights, builds a digest-pinned service image, SSHes to a GPU host, and
switches a Docker container after a JSON health check.

GPUStack already owns inference (vLLM, SGLang, scheduling, Higress). This
integration keeps model-deploy for **weight packaging** and hands **runtime**
to GPUStack via `source=local_path`.

Phase 1 ships adapters under [`contrib/model-deploy/`](../../contrib/model-deploy/README.md):
convert, plan, and apply. It does not rewrite the scheduler, worker, or
gateway.

## Mental model

```text
model-deploy ready.json / deploy.json / model-catalog.json
        │  convert
        ▼
GPUStack ModelCreate / deployment YAML
  source=local_path  backend=vLLM|SGLang
        │  schedule
        ▼
GPUStack Worker  ◀── NFS / rsync ──  /data/models/<model>/<version>
```

Same gates as `deploy.py plan` (preview, no side effects) and an explicit
apply. Jenkins / TOS / SSH docker switch stay outside GPUStack.

## What stays in model-deploy

| Stage | Owner |
|---|---|
| TOS upload + SHA256 manifest | model-deploy `upload` |
| Immutable `/data/models/<model>/<version>` | model-deploy download, NFS, or rsync — see [sync_weights.md](../../contrib/model-deploy/sync_weights.md) |
| Digest-pinned **custom** service image | model-deploy `build` (phase 2 / custom backend) |
| Jenkins `input` approval | Jenkins; gate `make model-deploy-apply` |
| Host GPU flock / port lock | Not mapped; GPUStack scheduler replaces it |
| Rollback previous container | GPUStack stop / start or re-import the previous YAML |

GPUStack **does not sync weights**. If the path is missing on a worker, the
instance will not become ready.

## Field mapping

| model-deploy | GPUStack |
|---|---|
| model name / `served_model_name` | `ModelCreate.name` |
| `/data/models/<model>/<version>` | `source=local_path`, `local_path=…` |
| catalog / `service.json` `runtime: vllm` | `backend=vLLM` |
| `runtime: sglang` | `backend=SGLang` |
| `python-adapter` / `custom-http` | Unsupported in phase 1 — see [Using Custom Inference Backends](../tutorials/using-custom-backends.md) |
| `runtime.gpus` (`"0,1"`) | `gpu_selector.gpu_ids` (`"worker:cuda:0"`) when `--worker` is set; otherwise auto-schedule |
| `runtime.environment` + catalog `environment` | `env` (offline flags such as `HF_HUB_OFFLINE=1` are kept) |
| `tensor_parallel_size` / `extra_args` | `backend_parameters` |
| `health_path` + `health_expect_json` | Post-apply probe in the bridge file; not a `ModelCreate` field |
| Stock vLLM / SGLang image | GPUStack-managed backend image (`image_name` unset) |
| Custom service image digest | Do not map to built-in vLLM; use a custom backend (phase 2) |

`gpu_ids` need the worker **advertise name**. Host index `0` is not a GPUStack
id by itself.

## Operator flow

1. **Prepare weights on workers.** Restore the immutable version to
   `/data/models/<model>/<version>` on every worker that may receive a replica
   ([sync checklist](../../contrib/model-deploy/sync_weights.md)).
2. **Convert** catalog + deploy JSON to a bridge document
   (`gpustack-release.json`) or a deployment YAML document.
3. **Plan** — print the mapping (name, `local_path`, backend, env keys,
   GPU notes). No API call.
4. **Apply** — `POST /v2/models` with `GPUSTACK_SERVER_URL` and
   `GPUSTACK_API_KEY`, or import the YAML in the UI. `--dry-run` prints the
   body and does not POST.
5. **Verify** — GPUStack UI (instance `Running`), playground, and the
   OpenAI-compatible `/v1/models` (or the catalog `health_path`). Compare
   `health_expect_json` yourself; GPUStack `Running` is not JSON equality.

```bash
make model-deploy-plan -- \
  --deploy tests/fixtures/model_deploy/vllm-standard/deploy.json \
  --catalog tests/fixtures/model_deploy/vllm-standard/model-catalog.json \
  --service tests/fixtures/model_deploy/vllm-standard/service.json \
  --model vllm-standard --version v1 --worker gpu-worker-a

make model-deploy-apply -- \
  --deploy tests/fixtures/model_deploy/vllm-standard/deploy.json \
  --catalog tests/fixtures/model_deploy/vllm-standard/model-catalog.json \
  --model vllm-standard --version v1 --dry-run
```

Export YAML without applying:

```bash
uv run python contrib/model-deploy/convert.py \
  --deploy … --catalog … --model vllm-standard --version v1 --format yaml
```

Import that document from **Models → Deployments** (export/import YAML), or
pass `--export-yaml path.yaml` to `apply.py`.

## Unsupported runtimes (phase 1)

`custom-http` and `python-adapter` (OCR, reranker adapters, and similar
templates) raise a clear error and point here. They are first-class in
model-deploy and are **not** built-in GPUStack backends. Packaging them as a
custom inference backend is a later change.

## Health and rollback

- Default probe for vLLM / SGLang: `GET /v1/models` and expect
  `{"object": "list"}` with a non-empty `data` when the catalog requires it.
- Rollback: stop the deployment, re-apply a previous bridge file, or import
  the previous YAML. GPUStack does not keep the last Docker container.

## Credentials and air-gap

- Placeholders only in-tree (`gpu-server.example.com`, `tos.example.com`).
- `GPUSTACK_API_KEY`, `TOS_*`, and registry passwords stay in the environment.
- Offline catalogs should keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  in `env`. A sample local-path catalog fragment is
  [`examples/catalog-local-path.yaml`](../../contrib/model-deploy/examples/catalog-local-path.yaml).

## Out of scope

Full TOS upload from GPUStack, SSH `docker run` replacement, a Jenkinsfile
inside GPUStack CI, and automatic weight sync to all workers.
