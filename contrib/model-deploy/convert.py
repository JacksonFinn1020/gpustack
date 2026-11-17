#!/usr/bin/env python3
"""Convert model-deploy catalog/deploy JSON to a GPUStack bridge document."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bridge import (
    ConvertError,
    add_convert_arguments,
    dump_deployment_yaml,
    dump_release_json,
    release_from_args,
    write_text,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert model-deploy catalog/deploy JSON to GPUStack "
            "ModelCreate JSON or deployment YAML (source=local_path)."
        )
    )
    add_convert_arguments(parser)
    parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        default="json",
        dest="fmt",
        help="json writes the bridge file; yaml writes a UI import document",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        release = release_from_args(args)
    except ConvertError as exc:
        logger.error("%s", exc)
        return 2
    text = (
        dump_deployment_yaml(release)
        if args.fmt == "yaml"
        else dump_release_json(release)
    )
    write_text(args.output, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
