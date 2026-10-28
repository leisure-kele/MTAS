"""Small offline checks for the public repository boundary.

This script deliberately avoids importing torch, loading data, or opening a
network connection.  It checks Python syntax, local relative-import closure,
and common private-path markers before a GitHub upload.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "atomseg"
PRIVATE_MARKERS = (
    "/Users/weason",
    "/personal/",
    "/data/wenxuan",
    "/mnt/xspark",
)


def python_files() -> list[Path]:
    return sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "supplementary").rglob("*.py")
    )


def main() -> None:
    files = python_files()
    modules = {path.stem for path in SOURCE.glob("*.py")}
    missing: list[str] = []
    private: list[str] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        for marker in PRIVATE_MARKERS:
            if marker in text:
                private.append(f"{path.relative_to(ROOT)}: {marker}")
        if path.parent != SOURCE:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module and node.module not in modules:
                    missing.append(
                        f"{path.relative_to(ROOT)} imports missing .{node.module}"
                    )

    if missing or private:
        details = "\n".join(missing + private)
        raise SystemExit(f"release verification failed:\n{details}")
    print(f"release verification passed ({len(files)} Python files)")


if __name__ == "__main__":
    main()
