"""Publish a local model directory as an immutable TOS release."""

import argparse
import json
import logging
import sys

from gpustack.routes.tos import TosIngestRequest, ingest_release

logger = logging.getLogger(__name__)


def setup_tos_ingest_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "tos-ingest",
        help="Upload a local model directory to TOS as an immutable release.",
        description=(
            "Scan a local directory (or an on-disk Hugging Face / ModelScope "
            "snapshot) and publish manifest.json plus content-addressed objects. "
            "Uses TOS_ACCESS_KEY / TOS_SECRET_KEY from the environment."
        ),
    )
    parser.add_argument("--local-path", required=True, help="Directory of model files.")
    parser.add_argument("--tos-endpoint", required=True)
    parser.add_argument("--tos-region", required=True)
    parser.add_argument("--tos-bucket", required=True)
    parser.add_argument("--tos-model", required=True)
    parser.add_argument("--tos-version", required=True)
    parser.add_argument(
        "--tos-release-path",
        default=None,
        help="Optional bucket prefix. Default is $TOS_PREFIX/<model>/<version>.",
    )
    parser.set_defaults(func=run)


def run(args):
    req = TosIngestRequest(
        local_path=args.local_path,
        tos_endpoint=args.tos_endpoint,
        tos_region=args.tos_region,
        tos_bucket=args.tos_bucket,
        tos_model=args.tos_model,
        tos_version=args.tos_version,
        tos_release_path=args.tos_release_path,
    )
    try:
        result = ingest_release(req)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
