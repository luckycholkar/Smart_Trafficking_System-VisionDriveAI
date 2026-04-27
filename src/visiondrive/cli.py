"""
CLI entrypoint.

Usage:
    visiondrive run                  # uvicorn + pipeline (production-like)
    visiondrive pipeline             # vision pipeline only (no API)
    visiondrive version
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from visiondrive import __version__
from visiondrive.settings import get_settings


def _cmd_run(_: argparse.Namespace) -> int:
    settings = get_settings()
    uvicorn.run(
        "visiondrive.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
    return 0


def _cmd_pipeline(_: argparse.Namespace) -> int:
    """Run the vision pipeline standalone, without the API/dashboard."""
    from visiondrive.core.data_bus import DataBus
    from visiondrive.core.pipeline import VisionPipeline
    from visiondrive.models import build_detector

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    data_bus = DataBus()
    detector = build_detector(settings.model)
    VisionPipeline(settings, detector, data_bus).run()
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"visiondrive {__version__}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visiondrive", description="VisionDrive AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start API + pipeline (uvicorn)").set_defaults(func=_cmd_run)
    sub.add_parser("pipeline", help="Run vision pipeline only").set_defaults(func=_cmd_pipeline)
    sub.add_parser("version", help="Print version").set_defaults(func=_cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
