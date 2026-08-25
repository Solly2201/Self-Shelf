"""Run the Self-Shelf dashboard server.

    python src/serve.py [--port 8765] [-n 50] [--data path.csv]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from selfshelf.webapp import DEFAULT_DATA, create_app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="self-shelf-dashboard",
        description="Self-Shelf pricing dashboard server",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "-n", "--items", type=int, default=50,
        help="number of test-set products to optimize (default: 50)",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="path to the retail dataset CSV",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        return 1
    app = create_app(data_path=str(args.data), num_items=args.items)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
