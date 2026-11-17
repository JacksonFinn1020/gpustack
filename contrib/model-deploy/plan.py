#!/usr/bin/env python3
"""Print a model-deploy → GPUStack plan without calling the API."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bridge import (
    ConvertError,
    add_convert_arguments,
    dump_release_json,
    format_plan,
    release_from_args,
    write_text,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview the GPUStack mapping. Does not upload or POST."
    )
    add_convert_arguments(parser)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="fmt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the text plan or bridge JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        release = release_from_args(args)
    except ConvertError as exc:
        logger.error("%s", exc)
        return 2
    text = dump_release_json(release) if args.fmt == "json" else format_plan(release)
    write_text(args.output, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
