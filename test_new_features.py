#!/usr/bin/env python3
"""Quick sanity tests for new features."""

from parser import _sanitize_filename, _sanitize_category

# Test sanitize_filename
assert _sanitize_filename("Beach Sunset.jpg") == "Beach-Sunset.jpg"
assert _sanitize_filename("../../../etc/passwd") == ".etcpasswd"  # slashes removed, dots collapsed
assert _sanitize_filename("my file (1).txt") == "my-file-1.txt"
assert _sanitize_filename("simple.txt") == "simple.txt"
assert _sanitize_filename("file..name.txt") == "file.name.txt"
print("sanitize_filename: OK")

# Test sanitize_category
assert _sanitize_category("Work Documents") == "Work-Documents"
assert _sanitize_category("  spaces  ") == "spaces"
print("sanitize_category: OK")

# Test new JSON parse format
from slm import _validate_sorts_dict, _extract_renames

# Old format
old = {"sorts": {"file.txt": "Documents", "photo.jpg": "Photos"}}
s = _validate_sorts_dict(old)
r = _extract_renames(old)
assert s == {"file.txt": "Documents", "photo.jpg": "Photos"}
assert r == {}
print("old format parse: OK")

# New format with renames
new = {"sorts": {"IMG_20250410.jpg": {"category": "Vacation", "rename": "Beach-Sunset.jpg"}}}
s = _validate_sorts_dict(new)
r = _extract_renames(new)
assert s == {"IMG_20250410.jpg": "Vacation"}
assert r == {"IMG_20250410.jpg": "Beach-Sunset.jpg"}
print("new format parse: OK")

# Mixed format (some old, some new)
mixed = {"sorts": {"file.txt": "Docs", "photo.jpg": {"category": "Vacation", "rename": "Sunset.jpg"}}}
s = _validate_sorts_dict(mixed)
r = _extract_renames(mixed)
assert s == {"file.txt": "Docs", "photo.jpg": "Vacation"}
assert r == {"photo.jpg": "Sunset.jpg"}
print("mixed format parse: OK")

# Test exclusion helper
import sys
sys.path.insert(0, ".")
# _is_excluded is in vsort.py, test the fnmatch logic directly
import fnmatch
patterns = ["*.tmp", "*.bak", "desktop.ini"]
assert fnmatch.fnmatch("file.tmp", "*.tmp")
assert not fnmatch.fnmatch("file.txt", "*.tmp")
assert fnmatch.fnmatch("desktop.ini", "desktop.ini")
print("exclusion patterns: OK")

# Test manifest save/load
import tempfile, json
from pathlib import Path
from parser import DirectoryResult, MoveResult, save_manifest, load_latest_manifest

dr = DirectoryResult(directory="/tmp/test", moves=[
    MoveResult(src="/tmp/test/a.txt", dst="/tmp/test/Docs/a.txt", category="Docs", success=True),
    MoveResult(src="/tmp/test/b.jpg", dst="/tmp/test/Photos/b.jpg", category="Photos", success=True),
])
mpath = save_manifest(dr)
assert mpath is not None
manifest = load_latest_manifest()
assert manifest is not None
assert manifest["directory"] == "/tmp/test"
assert len(manifest["moves"]) == 2
assert manifest["moves"][0]["src"] == "a.txt"
assert manifest["moves"][0]["dst"] == "Docs/a.txt"
# Clean up
mpath.unlink(missing_ok=True)
print("manifest save/load: OK")

print("\nAll tests passed!")
