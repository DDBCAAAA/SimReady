#!/usr/bin/env python3
"""mcmaster_catalog.py — Build a structured McMaster-Carr product catalog.

McMaster-Carr requires a login for all product browsing and blocks all
automated HTTP clients. This tool therefore works from a user-provided
list of part numbers collected manually during a logged-in browser session.

Workflow
--------
  Step 1 — Collect part numbers manually from www.mcmaster.com (logged in).
  Step 2 — Save them to a plain-text file (one per line) or a CSV.
  Step 3 — Run this script to produce a structured JSON + CSV catalog.
  Step 4 — Download STEP files manually from each product page.
  Step 5 — Convert to USD: python mcmaster_to_usd.py --part <PART> --file <FILE>

Usage
-----
    python mcmaster_catalog.py parts.txt
    python mcmaster_catalog.py parts.txt --tag motors --step-dir ~/Downloads
    python mcmaster_catalog.py parts.csv --tag bearings --out output/bearings

Input formats
-------------
Plain text (one part number per line):
    1718K33
    7014N106
    5972K394

CSV (with optional columns: part_number, name, url, category, notes):
    part_number,name,url,category,notes
    7014N106,Base-Mount AC Motor,https://www.mcmaster.com/7014N106/,Motors,

Output
------
    output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.json
    output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcmaster_catalog")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcmaster_catalog",
        description=(
            "Build a structured McMaster-Carr product catalog from a manually "
            "collected parts list. Outputs JSON + CSV with CAD/USD file status."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python mcmaster_catalog.py parts.txt\n"
            "  python mcmaster_catalog.py parts.txt --tag motors --step-dir ~/Downloads\n"
            "  python mcmaster_catalog.py parts.csv --tag bearings --out output/bearings\n"
        ),
    )
    p.add_argument(
        "parts_file", type=Path,
        help="Plain-text or CSV file of McMaster part numbers (one per line or CSV with 'part_number' column)",
    )
    p.add_argument(
        "--tag", "-t", default="catalog",
        help="Label used in output filenames, e.g. 'motors' (default: catalog)",
    )
    p.add_argument(
        "--step-dir", "-s", type=Path, default=None,
        help="Directory containing manually downloaded STEP files (searched recursively)",
    )
    p.add_argument(
        "--out", "-o", type=Path, default=Path("output/mcmaster_catalog"),
        help="Output directory (default: output/mcmaster_catalog)",
    )
    p.add_argument(
        "--no-api", action="store_true",
        help="Skip McMaster API enrichment even if credentials are configured",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from simready.acquisition.mcmaster_scraper import run_catalog

    try:
        json_path, csv_path = run_catalog(
            parts_file=args.parts_file,
            tag=args.tag,
            step_dir=args.step_dir,
            out_dir=args.out,
            use_api=not args.no_api,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\nCatalog saved:")
    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}\n")


if __name__ == "__main__":
    main()
