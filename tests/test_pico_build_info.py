"""Check the native companion's build stamp against the checkout that built it."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys


def git(root, *args):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    library_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "zig-out/lib/libPicoMlxServe.dylib"
    library = ctypes.CDLL(str(library_path.resolve()))
    build_info = library.pico_ms_build_info_v1
    build_info.argtypes = []
    build_info.restype = ctypes.c_char_p

    revision = git(root, "rev-parse", "--verify", "HEAD^{commit}")
    prefix = git(root, "rev-parse", "--show-prefix")
    payload = build_info()
    if revision.returncode != 0 or prefix.returncode != 0 or prefix.stdout.strip():
        assert payload is None, payload
        return

    expected = {"revision": revision.stdout.strip()}
    tag = git(root, "describe", "--tags", "--exact-match", expected["revision"])
    if tag.returncode == 0:
        expected["tag"] = tag.stdout.strip()
    status = git(root, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none")
    if status.returncode == 0:
        expected["dirty"] = bool(status.stdout.strip())

    actual = json.loads(payload)
    assert actual == expected, (actual, expected)
    print(f"Build info matches checkout: {actual}")


if __name__ == "__main__":
    main()
