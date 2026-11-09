"""Relative files under a READY ModelFile cache, matching archive ``paths``.

``POST /v2/tos/archives`` ``paths`` are POSIX relatives of the scan root:
the first ``resolved_paths`` entry when it is a directory; when it is a
file, that file's basename. Absolute paths, ``..``, and symlink escapes
are never returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.models import SourceEnum

MAX_LISTING_FILES = 5000
MAX_LISTING_DEPTH = 12
MAX_PER_PAGE = 200
DEFAULT_PER_PAGE = 100

_SKIP_DIR_NAMES = {".tos-tmp", ".git"}
_SKIP_FILE_NAMES = {
    "manifest.json",
    ".deployment-manifest.json",
    ".tos-archive.lock",
}


@dataclass
class RelativeFile:
    path: str
    size: int


@dataclass
class RelativeListing:
    items: List[RelativeFile] = field(default_factory=list)
    truncated: bool = False
    next_after: Optional[str] = None
    root_kind: str = "directory"


def archive_relative_ok(relative: str) -> bool:
    """True when ``relative`` is a legal ``archives.paths`` entry."""
    if not relative or not isinstance(relative, str):
        return False
    value = relative.replace("\\", "/")
    if value.startswith("/") or value.endswith("/"):
        return False
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    return True


def sanitize_archive_relative(relative: str) -> str:
    if not archive_relative_ok(relative):
        raise InvalidException(message=f"Illegal archive path: {relative}")
    return relative.replace("\\", "/")


def listing_allow_bases(cache_dir: Optional[str], model_file) -> List[Path]:
    """Directories a READY ModelFile may be listed from on this worker.

    Hugging Face / ModelScope / TOS stay under the worker ``cache_dir``
    (and ``local_dir`` when the download was redirected). Local Path
    stays under ``local_path`` / ``local_dir``. Filesystem root is never
    a base.
    """
    bases: List[Path] = []
    source = getattr(model_file, "source", None)
    source_value = source.value if hasattr(source, "value") else source
    local_dir = getattr(model_file, "local_dir", None)
    if source_value == SourceEnum.LOCAL_PATH.value:
        local_path = getattr(model_file, "local_path", None)
        if local_path:
            path = Path(local_path)
            bases.append(path if path.is_dir() else path.parent)
        if local_dir:
            bases.append(Path(local_dir))
    else:
        if cache_dir:
            bases.append(Path(cache_dir))
        if local_dir:
            bases.append(Path(local_dir))
    usable: List[Path] = []
    seen = set()
    for base in bases:
        resolved = _usable_listing_base(base)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        usable.append(resolved)
    return usable


def _usable_listing_base(base: Path) -> Optional[Path]:
    try:
        resolved = base.resolve()
    except OSError:
        return None
    if resolved == Path(resolved.anchor):
        return None
    return resolved


def assert_listing_root_allowed(
    resolved_paths: Sequence[str],
    *,
    cache_dir: Optional[str],
    model_file,
) -> Path:
    """Raise if the archive scan root is outside this ModelFile's cache."""
    root = archive_scan_root(resolved_paths)
    candidate = root.resolve()
    for base in listing_allow_bases(cache_dir, model_file):
        try:
            candidate.relative_to(base)
            return root
        except ValueError:
            continue
    raise PermissionError("listing root is outside the model cache")


def archive_scan_root(resolved_paths: Sequence[str]) -> Path:
    """Local path the archive uploader and this listing both start from."""
    if not resolved_paths:
        raise FileNotFoundError("READY model file has no resolved local path")
    first = Path(resolved_paths[0])
    if first.is_symlink():
        raise PermissionError("resolved path must not be a symlink")
    if first.is_dir() or first.is_file():
        return first
    raise FileNotFoundError(str(first))


def list_archive_relatives(
    resolved_paths: Sequence[str],
    *,
    after: Optional[str] = None,
    limit: int = DEFAULT_PER_PAGE,
) -> RelativeListing:
    """List files under the archive scan root, lexicographic, cursor-paged."""
    if limit < 1:
        limit = 1
    if limit > MAX_PER_PAGE:
        limit = MAX_PER_PAGE
    cursor = after.replace("\\", "/") if after else None
    if cursor is not None and not archive_relative_ok(cursor):
        raise InvalidException(message="after must be a relative archive path")

    root = archive_scan_root(resolved_paths)
    if root.is_file():
        rel = root.name
        listing = RelativeListing(root_kind="file")
        if archive_relative_ok(rel) and (cursor is None or rel > cursor):
            listing.items.append(RelativeFile(path=rel, size=root.stat().st_size))
        return listing

    collected: List[RelativeFile] = []
    hit_cap = False
    for rel, size in _walk_files_sorted(root):
        if len(collected) >= MAX_LISTING_FILES:
            hit_cap = True
            break
        collected.append(RelativeFile(path=rel, size=size))
    collected.sort(key=lambda item: item.path)
    page: List[RelativeFile] = []
    has_more = False
    for item in collected:
        if cursor is not None and item.path <= cursor:
            continue
        if len(page) == limit:
            has_more = True
            break
        page.append(item)
    truncated = has_more or hit_cap
    next_after = page[-1].path if page and truncated else None
    return RelativeListing(
        items=page,
        truncated=truncated,
        next_after=next_after,
        root_kind="directory",
    )


def filter_worker_listing_items(items: Iterable[dict]) -> List[RelativeFile]:
    """Drop untrusted worker rows; never forward absolute or escaping paths."""
    safe: List[RelativeFile] = []
    for raw in items:
        if not isinstance(raw, dict):
            path = getattr(raw, "path", None)
            size = getattr(raw, "size", None)
        else:
            path = raw.get("path")
            size = raw.get("size")
        if not archive_relative_ok(path):
            continue
        if type(size) is not int or size < 0:
            continue
        safe.append(RelativeFile(path=str(path).replace("\\", "/"), size=size))
    return safe


def _walk_files_sorted(
    root: Path, depth: int = 0, scan_root: Optional[Path] = None
) -> Iterator[tuple[str, int]]:
    scan_root = scan_root or root.resolve()
    try:
        entries = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if (
                    depth >= MAX_LISTING_DEPTH
                    or entry.name in _SKIP_DIR_NAMES
                    or entry.name.startswith(".")
                ):
                    continue
                yield from _walk_files_sorted(Path(entry.path), depth + 1, scan_root)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            if entry.name in _SKIP_FILE_NAMES:
                continue
            rel = Path(entry.path).resolve().relative_to(scan_root).as_posix()
            if not archive_relative_ok(rel):
                continue
            yield rel, entry.stat(follow_symlinks=False).st_size
        except (OSError, ValueError):
            continue
