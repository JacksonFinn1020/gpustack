#!/usr/bin/env python3
"""POST a converted ModelCreate body to GPUStack, or write import YAML."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from bridge import (
    ConvertError,
    add_convert_arguments,
    apply_body,
    dump_deployment_yaml,
    dump_release_json,
    load_json,
    release_from_args,
    write_text,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = "/v2/models"
# Management ModelCreate lives under HTTPClient.versioned_prefix (/v2), not
# the OpenAI-compatible GET /v1/models probe.
DRY_RUN_SERVER_PLACEHOLDER = "https://gpustack.example.com"
ENV_SERVER_URL = "GPUSTACK_SERVER_URL"
ENV_API_KEY = "GPUSTACK_API_KEY"

HttpPost = Callable[
    [str, Dict[str, str], Dict[str, Any], float],
    Tuple[int, Optional[Any], str],
]


class ApplyError(RuntimeError):
    """Apply failed; message must already be redacted."""


def redact(text: str, secret: Optional[str]) -> str:
    """Strip API keys and Bearer tokens from an error string."""
    redacted = text
    if secret:
        redacted = redacted.replace(secret, "***")
    return re.sub(r"(Bearer\s+)\S+", r"\1***", redacted, flags=re.IGNORECASE)


def default_http_post(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: float,
) -> Tuple[int, Optional[Any], str]:
    """httpx POST used when no transport is injected."""
    import httpx

    response = httpx.post(url, headers=headers, json=body, timeout=timeout)
    payload: Optional[Any]
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload, response.text


def _require_apply_credentials(server_url: str, api_key: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not server_url:
        raise ApplyError(f"{ENV_SERVER_URL} is required")
    if not api_key:
        raise ApplyError(f"{ENV_API_KEY} is required")


def _model_create_body(release: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return apply_body(release)
    except ConvertError as exc:
        raise ApplyError(str(exc)) from exc


def apply_release(
    release: Dict[str, Any],
    server_url: str,
    api_key: str,
    *,
    dry_run: bool = False,
    models_path: str = DEFAULT_MODELS_PATH,
    timeout: float = 30.0,
    http_post: Optional[HttpPost] = None,
) -> Dict[str, Any]:
    """POST ``model_create`` to ``{server}{models_path}``.

    Args:
        release: Bridge document from convert.
        server_url: GPUStack base URL (no path). Optional for dry-run.
        api_key: Bearer token. Not required for dry-run.
        dry_run: When true, return the intended request and do not POST.
        models_path: API path (default ``/v2/models``).
        timeout: httpx timeout seconds.
        http_post: Optional ``(url, headers, body, timeout) -> (status, json, text)``.

    Returns:
        API JSON, or a dry-run description.
    """
    _require_apply_credentials(server_url, api_key, dry_run)
    body = _model_create_body(release)
    base = server_url.rstrip("/") if server_url else DRY_RUN_SERVER_PLACEHOLDER
    url = base + models_path
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if dry_run:
        logger.info("dry-run: would POST %s", url)
        return {"dry_run": True, "method": "POST", "url": url, "body": body}
    poster = http_post or default_http_post
    try:
        status, payload, text = poster(url, headers, body, timeout)
    except ApplyError:
        raise
    except Exception as exc:
        raise ApplyError(redact(str(exc), api_key)) from exc
    if status < 200 or status >= 300:
        raise ApplyError(redact(f"GPUStack API returned {status}: {text}", api_key))
    if isinstance(payload, dict):
        return payload
    return {"status": status, "body": payload}


def load_release(args: argparse.Namespace) -> Dict[str, Any]:
    """Load ``--spec`` or convert from catalog/deploy flags."""
    if args.spec:
        return load_json(args.spec)
    return release_from_args(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a converted spec to GPUStack POST /v2/models. "
            "Use --dry-run to print the body without HTTP."
        )
    )
    add_convert_arguments(parser)
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Existing gpustack-release JSON (skips --deploy/--catalog)",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get(ENV_SERVER_URL, ""),
        dest="server_url",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(ENV_API_KEY, ""),
        dest="api_key",
        help=f"Or set {ENV_API_KEY} (never commit this value)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--models-path",
        default=DEFAULT_MODELS_PATH,
        dest="models_path",
        help="API path, default /v2/models",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--export-yaml",
        type=Path,
        default=None,
        dest="export_yaml",
        help="Write a deployment YAML document (UI import) and still apply unless --dry-run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON result (dry-run request or API response)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        release = load_release(args)
        if args.export_yaml:
            write_text(args.export_yaml, dump_deployment_yaml(release))
        result = apply_release(
            release,
            args.server_url,
            args.api_key,
            dry_run=args.dry_run,
            models_path=args.models_path,
            timeout=args.timeout,
        )
    except (ConvertError, ApplyError) as exc:
        logger.error("%s", redact(str(exc), args.api_key))
        return 2
    text = dump_release_json(result) if isinstance(result, dict) else json.dumps(result)
    write_text(args.output, text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
