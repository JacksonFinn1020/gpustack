# model-deploy → GPUStack adapter (phase 1)

Convert a model-deploy catalog/deploy pair into a GPUStack `local_path`
deployment. Serving stays in GPUStack (scheduler, replicas, Higress). Weight
packaging and TOS upload stay in model-deploy.

Operator flow: **weights on workers → convert → plan → apply → verify**.

Docs: [Integrate with model-deploy](../../docs/integrations/integrate-with-model-deploy.md).
Schema: [schema.md](schema.md). Weight sync: [sync_weights.md](sync_weights.md).

## Commands

From the repository root (after `make deps`):

```bash
# Human-readable plan; does not call the API.
uv run python contrib/model-deploy/plan.py \
  --deploy tests/fixtures/model_deploy/vllm-standard/deploy.json \
  --catalog tests/fixtures/model_deploy/vllm-standard/model-catalog.json \
  --service tests/fixtures/model_deploy/vllm-standard/service.json \
  --model vllm-standard \
  --version v1 \
  --worker gpu-worker-a

# Bridge JSON or deployment YAML on stdout.
uv run python contrib/model-deploy/convert.py \
  --deploy tests/fixtures/model_deploy/vllm-standard/deploy.json \
  --catalog tests/fixtures/model_deploy/vllm-standard/model-catalog.json \
  --service tests/fixtures/model_deploy/vllm-standard/service.json \
  --model vllm-standard \
  --version v1 \
  --format json

# Dry-run apply (no HTTP, credentials optional). Real apply needs
# GPUSTACK_SERVER_URL + GPUSTACK_API_KEY and POSTs /v2/models
# (management ModelCreate; GET /v1/models is the OpenAI probe).
uv run python contrib/model-deploy/apply.py \
  --deploy tests/fixtures/model_deploy/vllm-standard/deploy.json \
  --catalog tests/fixtures/model_deploy/vllm-standard/model-catalog.json \
  --model vllm-standard \
  --version v1 \
  --dry-run
```

Makefile wrappers (same flags after the target):

```bash
make model-deploy-plan -- --deploy … --catalog … --model vllm-standard --version v1
make model-deploy-apply -- --spec contrib/model-deploy/examples/vllm-bridge.json --dry-run
```

## Phase 1 backends

| model-deploy `runtime` | GPUStack `backend` |
|---|---|
| `vllm` | `vLLM` |
| `sglang` | `SGLang` |
| `custom-http`, `python-adapter` | **error** — see the integration doc |

## Credentials

Examples use placeholders only (`gpu-server.example.com`, `tos.example.com`).
Set `GPUSTACK_API_KEY` in the environment; never commit it.
