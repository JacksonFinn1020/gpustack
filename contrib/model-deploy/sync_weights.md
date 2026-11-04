# Sync weights onto GPUStack workers

GPUStack does **not** download model-deploy TOS manifests and does **not**
copy weights between workers. `source=local_path` only points at a directory
that must already exist on each scheduled worker.

## Layout

model-deploy writes (or restores) an immutable version directory:

```text
/data/models/<model>/<version>/
```

The converter sets `local_path` to that path. Every replica that the
scheduler may place must see the same files (including `config.json` when the
catalog lists it in `required_files`).

## Operator checklist

1. Finish model-deploy `upload` (TOS + SHA256 manifest) for the version you
   will serve. Do not overwrite an existing version; bump the version instead.
2. On each target GPU worker, restore that version to
   `/data/models/<model>/<version>`:
   - Shared mount: NFS / PVC already covering `/data/models`.
   - Per-node copy: model-deploy remote download, or `rsync -a` from a
     bastion that already has the tree.
3. Confirm the path on the worker (not only on the server):

   ```bash
   ls /data/models/<model>/<version>/config.json
   ```

4. If only some workers have the tree, set a GPUStack worker selector (or
   `--worker` so `gpu_selector.gpu_ids` pin placement). Multi-worker replicas
   without a shared filesystem will fail scheduling or start.
5. Then run convert → plan → apply. Do not apply first and copy weights later
   unless you keep `replicas=0` until the path exists.

## What this adapter will not do

- Call TOS / Volcengine APIs.
- SSH as `model-deploy` to pull objects.
- Replicate a directory to every worker after apply.

Keep those steps in model-deploy, Jenkins, or your storage platform.
