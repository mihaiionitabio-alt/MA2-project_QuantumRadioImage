"""Backward-compatible entry point; use :mod:`run_eht2019` for new runs."""

from run_eht2019 import main


if __name__ == "__main__":
    raise SystemExit(main())
