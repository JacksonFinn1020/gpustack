"""Map model-deploy catalog/deploy JSON onto a GPUStack ModelCreate body."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SOURCE_KIND = "model-deploy"
DEFAULT_MODEL_ROOT = "/data/models"
PHASE1_DOC = "docs/integrations/integrate-with-model-deploy.md"
CUSTOM_BACKEND_DOC = "docs/tutorials/using-custom-backends.md"

SUPPORTED_RUNTIMES = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}
UNSUPPORTED_RUNTIMES = ("custom-http", "python-adapter")
RUNTIME_ALIASES = tuple(
    sorted(
        (*SUPPORTED_RUNTIMES.keys(), *UNSUPPORTED_RUNTIMES),
        key=len,
        reverse=True,
    )
)

TP_FLAGS = {
    "vllm": "--tensor-parallel-size={n}",
    "sglang": "--tp={n}",
}

APPLY_OMIT_FIELDS = frozenset({"cluster_name"})


class ConvertError(ValueError):
    """Invalid or incomplete model-deploy input."""


class UnsupportedRuntimeError(ConvertError):
    """Launcher that phase 1 must not silently map to a built-in backend."""

    def __init__(self, runtime: str) -> None:
        self.runtime = runtime
        super().__init__(
            f"model-deploy runtime {runtime!r} is unsupported in phase 1 "
            f"(only vllm and sglang). Use a custom inference backend; see "
            f"{PHASE1_DOC} and {CUSTOM_BACKEND_DOC}."
        )


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from ``path``.

    Args:
        path: File to read.

    Returns:
        Parsed object.

    Raises:
        ConvertError: File missing, not an object, or invalid JSON.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConvertError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConvertError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConvertError(f"{path} must contain a JSON object")
    return data


def resolve_catalog_entry(
    catalog: Dict[str, Any], model: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """Pick one catalog profile.

    Args:
        catalog: model-catalog.json object (name → profile).
        model: Catalog key. Required when the catalog has multiple entries.

    Returns:
        ``(key, profile)``.
    """
    if model:
        profile = catalog.get(model)
        if not isinstance(profile, dict):
            raise ConvertError(f"catalog has no profile {model!r}")
        return model, profile
    keys = [key for key, value in catalog.items() if isinstance(value, dict)]
    if len(keys) != 1:
        raise ConvertError("catalog has multiple models; pass --model <catalog-key>")
    return keys[0], catalog[keys[0]]


def infer_runtime(
    model_key: str,
    profile: Dict[str, Any],
    service: Optional[Dict[str, Any]],
    explicit: Optional[str],
) -> str:
    """Resolve the model-deploy launcher name.

    Order: ``--runtime``, ``service.json`` ``runtime``, catalog ``runtime``,
    then a catalog-key prefix (``vllm-standard`` → ``vllm``).
    """
    for candidate in (
        explicit,
        (service or {}).get("runtime"),
        profile.get("runtime"),
        _runtime_from_key(model_key),
    ):
        if candidate:
            return str(candidate).strip().lower()
    raise ConvertError(
        f"cannot infer runtime for {model_key!r}; pass --runtime or service.json"
    )


def _runtime_from_key(model_key: str) -> Optional[str]:
    key = model_key.strip().lower()
    for name in RUNTIME_ALIASES:
        if key == name or key.startswith(f"{name}-"):
            return name
    return None


def map_backend(runtime: str) -> str:
    """Return the GPUStack backend name or raise for unsupported launchers."""
    if runtime in UNSUPPORTED_RUNTIMES:
        raise UnsupportedRuntimeError(runtime)
    backend = SUPPORTED_RUNTIMES.get(runtime)
    if backend is None:
        raise UnsupportedRuntimeError(runtime)
    return backend


def merge_env(*layers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Merge environment maps; later layers win. Values are stringified."""
    merged: Dict[str, str] = {}
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            if value is None:
                continue
            merged[str(key)] = str(value)
    return merged


def weight_local_path(model_root: str, model: str, version: str) -> str:
    """Build ``{model_root}/{model}/{version}`` with POSIX separators."""
    root = model_root.rstrip("/") or DEFAULT_MODEL_ROOT
    return f"{root}/{model}/{version}"


def parse_gpu_indices(gpus: str) -> List[str]:
    """Split ``runtime.gpus`` (``0`` or ``0,1``) into index strings."""
    return [part.strip() for part in gpus.split(",") if part.strip()]


def build_gpu_selector(
    worker: Optional[str],
    gpus: Optional[str],
    device_type: str,
) -> Optional[Dict[str, Any]]:
    """Map host GPU indices to GPUStack ``gpu_ids`` when a worker is known."""
    if not worker or not gpus:
        return None
    gpu_ids = [f"{worker}:{device_type}:{index}" for index in parse_gpu_indices(gpus)]
    if not gpu_ids:
        return None
    return {"gpu_ids": gpu_ids}


def build_backend_parameters(
    runtime: str,
    service: Optional[Dict[str, Any]],
) -> List[str]:
    """TP flag plus ``service.json`` ``extra_args``."""
    params: List[str] = []
    service = service or {}
    tp = service.get("tensor_parallel_size")
    if isinstance(tp, int) and tp > 1:
        template = TP_FLAGS.get(runtime)
        if template:
            params.append(template.format(n=tp))
    extra = service.get("extra_args") or []
    if isinstance(extra, list):
        params.extend(str(item) for item in extra)
    return params


def convert_release(
    deploy: Dict[str, Any],
    catalog: Dict[str, Any],
    *,
    service: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    version: str,
    worker: Optional[str] = None,
    device_type: str = "cuda",
    runtime: Optional[str] = None,
    cluster_name: Optional[str] = None,
    replicas: int = 1,
) -> Dict[str, Any]:
    """Build a ``gpustack-release`` object from model-deploy inputs.

    Args:
        deploy: Deploy / environment JSON.
        catalog: ``model-catalog.json`` object.
        service: Optional ``service.json``.
        model: Catalog key.
        version: Immutable weight version.
        worker: GPUStack worker advertise name for ``gpu_selector``.
        device_type: Device token in ``gpu_ids`` (default ``cuda``).
        runtime: Override inferred launcher.
        cluster_name: Optional YAML-import cluster name.
        replicas: Deployment replica count.

    Returns:
        Bridge document (schema version 1).
    """
    if not version:
        raise ConvertError("version is required (immutable weight directory)")
    model_key, profile = resolve_catalog_entry(catalog, model)
    runtime_name = infer_runtime(model_key, profile, service, runtime)
    backend = map_backend(runtime_name)
    server = deploy.get("server") or {}
    runtime_cfg = deploy.get("runtime") or {}
    model_root = str(server.get("model_root") or DEFAULT_MODEL_ROOT)
    served = _served_name(model_key, profile, service)
    local_path = weight_local_path(model_root, model_key, version)
    env = merge_env(
        profile.get("environment"),
        runtime_cfg.get("environment"),
        ((service or {}).get("custom") or {}).get("environment"),
    )
    gpu_selector = build_gpu_selector(worker, runtime_cfg.get("gpus"), device_type)
    model_create: Dict[str, Any] = {
        "name": served,
        "source": "local_path",
        "local_path": local_path,
        "backend": backend,
        "backend_parameters": build_backend_parameters(runtime_name, service),
        "env": env,
        "replicas": replicas,
        "categories": ["llm"],
        "enable_model_route": True,
        "description": (
            f"Converted from model-deploy {model_key}@{version} ({runtime_name})"
        ),
    }
    if gpu_selector:
        model_create["gpu_selector"] = gpu_selector
    if cluster_name:
        model_create["cluster_name"] = cluster_name
    notes = _notes(worker, runtime_cfg.get("gpus"), local_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "model_deploy": {
            "model": model_key,
            "version": version,
            "runtime": runtime_name,
            "served_model_name": served,
            "model_root": model_root,
            "physical_gpus": str(runtime_cfg.get("gpus") or ""),
        },
        "model_create": model_create,
        "probe": _probe(profile, service),
        "notes": notes,
    }


def _served_name(
    model_key: str,
    profile: Dict[str, Any],
    service: Optional[Dict[str, Any]],
) -> str:
    env = profile.get("environment") or {}
    for candidate in (
        (service or {}).get("served_model_name"),
        (service or {}).get("name"),
        env.get("MODEL_NAME"),
        model_key,
    ):
        if candidate:
            return str(candidate)
    return model_key


def _probe(
    profile: Dict[str, Any], service: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    health = (service or {}).get("health") or {}
    return {
        "health_path": health.get("path") or profile.get("health_path") or "",
        "health_expect_json": health.get("expect_json")
        or profile.get("health_expect_json")
        or {},
        "health_require_nonempty_data": bool(
            health.get("require_nonempty_data")
            if "require_nonempty_data" in health
            else profile.get("health_require_nonempty_data")
        ),
    }


def _notes(worker: Optional[str], gpus: Optional[str], local_path: str) -> List[str]:
    notes = [
        f"Place immutable weights at {local_path} on every target worker "
        "(NFS/rsync/model-deploy download). GPUStack does not sync TOS.",
        "After apply, probe health_path yourself; GPUStack instance "
        f"Running is not model-deploy JSON equality (see {PHASE1_DOC}).",
    ]
    if gpus and not worker:
        notes.append(
            f"runtime.gpus={gpus!r} is a host index; pass --worker "
            "<advertise-name> to set gpu_selector, or leave unset for "
            "auto-schedule."
        )
    return notes


def apply_body(release: Dict[str, Any]) -> Dict[str, Any]:
    """``model_create`` without YAML-only keys for ``POST /v2/models``."""
    body = dict(release.get("model_create") or {})
    for key in APPLY_OMIT_FIELDS:
        body.pop(key, None)
    missing = [
        name
        for name in ("name", "source", "local_path", "backend")
        if not body.get(name)
    ]
    if missing:
        raise ConvertError("spec model_create missing " + ", ".join(missing))
    if body.get("source") != "local_path":
        raise ConvertError("spec model_create.source must be local_path")
    return body


def format_plan(release: Dict[str, Any]) -> str:
    """Human-readable plan (no API call)."""
    create = release.get("model_create") or {}
    provenance = release.get("model_deploy") or {}
    probe = release.get("probe") or {}
    env = create.get("env") or {}
    selector = create.get("gpu_selector") or {}
    lines = [
        "model-deploy → GPUStack plan (no API call)",
        f"  model:            {provenance.get('model')}",
        f"  version:          {provenance.get('version')}",
        f"  served_name:      {create.get('name')}",
        f"  local_path:       {create.get('local_path')}",
        f"  backend:          {create.get('backend')}",
        f"  runtime:          {provenance.get('runtime')}",
        f"  replicas:         {create.get('replicas')}",
        f"  backend_parameters: {create.get('backend_parameters') or []}",
        f"  env_keys:         {sorted(env)}",
        f"  gpu_selector:     {selector or '(unset — auto-schedule)'}",
        f"  physical_gpus:    {provenance.get('physical_gpus') or '(none)'}",
        f"  probe.path:       {probe.get('health_path')}",
        f"  probe.expect:     {probe.get('health_expect_json')}",
    ]
    for note in release.get("notes") or []:
        lines.append(f"  note: {note}")
    return "\n".join(lines) + "\n"


def dump_release_json(release: Dict[str, Any]) -> str:
    return json.dumps(release, indent=2, ensure_ascii=False) + "\n"


def dump_deployment_yaml(release: Dict[str, Any]) -> str:
    """Single-entry deployment document for UI import."""
    import yaml

    header = (
        "# Generated by contrib/model-deploy (not a live GPUStack export)\n"
        "# Import from Models → Deployments, or POST model_create via apply.py.\n"
    )
    return header + yaml.safe_dump(
        dict(release.get("model_create") or {}),
        sort_keys=False,
        allow_unicode=True,
    )


def add_convert_arguments(parser: Any) -> None:
    """Shared CLI flags for convert / plan / apply."""
    parser.add_argument("--deploy", type=Path, help="Deploy JSON path")
    parser.add_argument("--catalog", type=Path, help="model-catalog.json path")
    parser.add_argument("--service", type=Path, default=None, help="service.json")
    parser.add_argument("--model", default=None, help="Catalog key")
    parser.add_argument("--version", default=None, help="Weight version")
    parser.add_argument("--worker", default=None, help="GPUStack worker name")
    parser.add_argument("--device-type", default="cuda", dest="device_type")
    parser.add_argument("--runtime", default=None, help="Override launcher")
    parser.add_argument("--cluster-name", default=None, dest="cluster_name")
    parser.add_argument("--replicas", type=int, default=1)


def release_from_args(args: Any) -> Dict[str, Any]:
    """Load files from parsed convert args and convert."""
    if not args.deploy or not args.catalog:
        raise ConvertError("--deploy and --catalog are required")
    if not args.version:
        raise ConvertError("--version is required")
    service = load_json(args.service) if args.service else None
    return convert_release(
        load_json(args.deploy),
        load_json(args.catalog),
        service=service,
        model=args.model,
        version=args.version,
        worker=args.worker,
        device_type=args.device_type,
        runtime=args.runtime,
        cluster_name=args.cluster_name,
        replicas=args.replicas,
    )


def write_text(path: Optional[Path], text: str) -> None:
    if path is None:
        print(text, end="")
        return
    path.write_text(text, encoding="utf-8")
    logger.info("wrote %s", path)


def required_release_fields(release: Dict[str, Any]) -> Sequence[str]:
    """Fields a unit test (or caller) can treat as the documented contract."""
    create = release.get("model_create") or {}
    missing = [
        name
        for name in (
            "schema_version",
            "source_kind",
            "model_deploy",
            "model_create",
        )
        if name not in release
    ]
    missing.extend(
        f"model_create.{name}"
        for name in ("name", "source", "local_path", "backend")
        if name not in create
    )
    return missing
