"""Allow `python -m visiondrive ...`."""

from visiondrive.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
