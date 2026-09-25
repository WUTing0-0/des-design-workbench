"""Create the self-contained public workbench archive."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_assets"
NAME = "DES_Design_Workbench_Portable_v0.2.0"

INCLUDE = [
    "dist", "examples/thymol_octanoic_acid", "portable_data", "portable_runtime",
    "runtime_models", "local_server.py", "property_inference.py",
    "requirements-portable.txt", "INSTALL_AND_START_WINDOWS.bat",
    "install_and_start_linux.sh", "PORTABLE_README_CN.md", "MODEL_MANIFEST.json",
]


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / f"{NAME}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in INCLUDE:
            source = ROOT / rel
            files = source.rglob("*") if source.is_dir() else [source]
            for path in files:
                if path.is_file() and "__pycache__" not in path.parts:
                    arc = Path(NAME) / path.relative_to(ROOT)
                    zf.write(path, arc.as_posix())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"file": target.name, "sha256": digest, "bytes": target.stat().st_size}
    (OUT / f"{NAME}.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
