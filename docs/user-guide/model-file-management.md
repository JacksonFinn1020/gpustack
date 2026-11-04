# Model File Management

GPUStack allows admins to download and manage model files.

## Add Model File

GPUStack currently supports models from [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn), local paths, and Volcengine TOS. To add model files, navigate to the `Model Files` page. The TOS source can also be created by deploying a model with `source: tos`; the assigned worker then pulls `manifest.json` and verifies each object.

### Add a Hugging Face Model

1. Click the `Add Model File` button and select `Hugging Face` from the dropdown.
2. Use the search bar in the top left to find a model by name, e.g., `Qwen/Qwen3-0.6B`.
3. _(Optional)_ For GGUF models, select the desired quantization format from `Available Files`.
4. Select the target worker to download the model file.
5. _(Optional)_ Specify a `Local Directory` to download the model to a custom path instead of the GPUStack cache directory.
6. Click the `Save` button.

### Add a ModelScope Model

1. Click the `Add Model File` button and select `ModelScope` from the dropdown.
2. Use the search bar in the top left to find a model by name, e.g., `Qwen/Qwen3-0.6B`.
3. _(Optional)_ For GGUF models, select the desired quantization format from `Available Files`.
4. Select the target worker to download the model file.
5. _(Optional)_ Specify a `Local Directory` to download the model to a custom path instead of the GPUStack cache directory.
6. Click the `Save` button.

### Add a Local Path Model

You can add models from a local path. The path can be a directory (e.g., a Hugging Face model folder) or a file (e.g., a GGUF model) located on the worker.

1. Click the `Add Model File` button and select `Local Path` from the dropdown.
2. Enter the `Model Path`.
3. Select the target worker.
4. Click the `Save` button.

### Add a TOS Model

Ingest the release first (`gpustack tos-ingest` or `POST /v2/tos/releases`), then create a model file (or a deployment that creates one) with `source: tos` and the TOS coordinate fields documented in [Model Deployment Management](model-deployment-management.md#deploying-a-tos-model). Select the target worker. GPUStack downloads the immutable release onto that worker; credentials come from `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` on the worker, not from the API payload. Workers need `pip install tos`.

## Retry Download

If a model file download fails — or gets stuck at a very low download speed — you can retry it:

1. Navigate to the `Model Files` page.
2. Locate the model file.
3. Click the ellipsis button in the `Operations` column and select `Retry Download`.
4. GPUStack will attempt to download the model file again from the specified source.

## Deploy Model

Models can be deployed from model files. Since the model is stored on a specific worker, GPUStack will add a worker selector using the `worker-name` key to ensure proper scheduling.

!!! tip

    If you want a model to fail over across nodes, make sure all nodes in the cluster can access the model files from the same path, and manually remove the `worker-name` label from the worker selector.

1. Navigate to the `Model Files` page.
2. Find the model file you want to deploy.
3. Click the `Deploy` button in the `Operations` column.
4. Review or adjust the `Name`, `Backend`, `Backend Version`, `Replicas`, and other deployment parameters.
5. Click the `Save` button.

## Delete Model File

1. Navigate to the `Model Files` page.
2. Find the model file you want to delete.
3. Click the ellipsis button in the `Operations` column and select `Delete`.
4. _(Optional)_ Check the `Also delete the file from disk` option.
5. Click the `Delete` button to confirm.
