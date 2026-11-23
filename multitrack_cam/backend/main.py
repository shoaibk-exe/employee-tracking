"""CLI entrypoint for the MTMCT API server."""

from __future__ import annotations

import argparse
import os

import uvicorn

from backend.api.app import create_app
from backend.config import load_app_config


def main() -> None:
    parser = argparse.ArgumentParser(description="MTMCT multi-camera tracking server")
    parser.add_argument("--settings", default=os.environ.get("MTMCT_SETTINGS", "config/settings.yaml"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_app_config(args.settings)
    host = args.host or os.environ.get("MTMCT_API_HOST") or str(config.get("api", "host", default="0.0.0.0"))
    port = args.port or int(os.environ.get("MTMCT_API_PORT") or config.get("api", "port", default=8090))

    app = create_app(args.settings)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
