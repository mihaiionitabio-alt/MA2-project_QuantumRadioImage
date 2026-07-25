"""Run the official EHT 2019 M87 imaging pipeline on Windows or Linux.

The official pipeline is executed from its downloaded source.  Its Fourier
backend is replaced in memory with the value of ``EHT_FT_BACKEND`` (``direct``
by default), so the third-party checkout remains unmodified.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
PIPELINE = (
    ROOT
    / "external"
    / "eht-m87-imaging"
    / "eht-imaging"
    / "eht-imaging_pipeline.py"
)


def _install_windows_resource_shim() -> None:
    """Provide the Unix memory-reporting API used by paramsurvey on Windows."""
    if os.name != "nt":
        return
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0

    def getrusage(_: int) -> tuple[int, ...]:
        resident_bytes = psutil.Process().memory_info().rss
        return (0, 0, resident_bytes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    resource.getrusage = getrusage
    sys.modules.setdefault("resource", resource)


def run_official_pipeline() -> None:
    if not PIPELINE.exists():
        raise FileNotFoundError(
            f"Official pipeline not found at {PIPELINE}. "
            "Run prepare_eht_data.py first."
        )

    backend = os.environ.get("EHT_FT_BACKEND", "direct").lower()
    if backend not in {"direct", "fast", "nfft"}:
        raise ValueError("EHT_FT_BACKEND must be direct, fast, or nfft.")

    source = PIPELINE.read_text(encoding="utf-8")
    original = "ttype     = 'nfft'"
    replacement = f"ttype     = '{backend}'"
    if source.count(original) != 1:
        raise RuntimeError(
            "Could not locate the official pipeline's Fourier-backend setting. "
            "Check that data product 2019-D01-02 is installed unmodified."
        )
    source = source.replace(original, replacement, 1)
    _install_windows_resource_shim()
    namespace = {
        "__name__": "__main__",
        "__file__": str(PIPELINE),
        "__package__": None,
    }
    exec(compile(source, str(PIPELINE), "exec"), namespace)


def main() -> int:
    run_official_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
