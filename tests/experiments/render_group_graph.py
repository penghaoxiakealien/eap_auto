#!/usr/bin/env python3
"""Render a grouped graph JSON payload into a PNG diagram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_grouped_graph import draw_group_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render grouped graph JSON to PNG")
    parser.add_argument("input_json", type=Path, help="Path to grouped graph JSON")
    parser.add_argument(
        "--output-png",
        type=Path,
        help="Optional PNG output path; defaults to input name with .png suffix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = json.loads(args.input_json.read_text())
    output_path = args.output_png or args.input_json.with_suffix(".png")

    draw_group_graph(data, output_path)


if __name__ == "__main__":
    main()
