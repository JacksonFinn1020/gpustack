from pathlib import Path

import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.utils.model_file_listing import (
    archive_relative_ok,
    filter_worker_listing_items,
    list_archive_relatives,
)


def test_relative_path_rejects_absolute_and_dotdot():
    assert archive_relative_ok("weights/model.gguf")
    assert not archive_relative_ok("/etc/passwd")
    assert not archive_relative_ok("../escape")
    assert not archive_relative_ok("foo/../bar")
    assert not archive_relative_ok("")


def test_directory_listing_nested_gguf(tmp_path):
    root = tmp_path / "cache"
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_bytes(b"{}")
    gguf = root / "weights" / "Llama-3-8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    (root / ".tos-tmp").mkdir()
    (root / ".tos-tmp" / "partial").write_bytes(b"x")
    listing = list_archive_relatives([str(root)])
    paths = [item.path for item in listing.items]
    assert paths == ["config.json", "weights/Llama-3-8B-Q4_K_M.gguf"]
    assert listing.root_kind == "directory"
    q4 = [item for item in listing.items if item.path.endswith(".gguf")][0]
    assert q4.size == 4


def test_single_file_resolved_path_uses_basename(tmp_path):
    gguf = tmp_path / "Llama-3-8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    listing = list_archive_relatives([str(gguf)])
    assert listing.root_kind == "file"
    assert [item.path for item in listing.items] == ["Llama-3-8B-Q4_K_M.gguf"]


def test_symlink_escape_is_not_listed(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "ok.bin").write_bytes(b"ok")
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")
    (root / "link.bin").symlink_to(outside)
    listing = list_archive_relatives([str(root)])
    assert [item.path for item in listing.items] == ["ok.bin"]


def test_missing_directory_raises(tmp_path):
    missing = tmp_path / "gone"
    with pytest.raises(FileNotFoundError):
        list_archive_relatives([str(missing)])


def test_listing_is_lexicographic_across_nested_dirs(tmp_path):
    root = tmp_path / "cache"
    (root / "b").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a")
    (root / "c.bin").write_bytes(b"c")
    (root / "b" / "x.bin").write_bytes(b"x")
    listing = list_archive_relatives([str(root)], limit=2)
    assert [item.path for item in listing.items] == ["a.bin", "b/x.bin"]
    assert listing.truncated is True
    rest = list_archive_relatives([str(root)], after=listing.next_after, limit=2)
    assert [item.path for item in rest.items] == ["c.bin"]


def test_cursor_pagination(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    for name in ("a.bin", "b.bin", "c.bin"):
        (root / name).write_bytes(b"x")
    first = list_archive_relatives([str(root)], limit=2)
    assert [item.path for item in first.items] == ["a.bin", "b.bin"]
    assert first.truncated is True
    assert first.next_after == "b.bin"
    rest = list_archive_relatives([str(root)], after=first.next_after, limit=2)
    assert [item.path for item in rest.items] == ["c.bin"]
    assert rest.truncated is False


def test_untrusted_worker_rows_are_dropped():
    safe = filter_worker_listing_items(
        [
            {"path": "/etc/passwd", "size": 1},
            {"path": "../escape", "size": 1},
            {"path": "weights/model.gguf", "size": 9},
            {"path": "ok.bin", "size": -1},
        ]
    )
    assert [item.path for item in safe] == ["weights/model.gguf"]


def test_illegal_after_rejected():
    with pytest.raises(InvalidException):
        list_archive_relatives(["/tmp"], after="/etc")
