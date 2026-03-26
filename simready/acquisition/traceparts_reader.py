"""traceparts_reader.py — Parse TraceParts CAD folder structure.

Each part folder under data/TraceParts/ contains:
  - A .stp file (the CAD geometry)
  - A .txt file (semicolon-delimited spec sheet)

The .txt format is:
    "Symbol";"Value";"Unit";
    "REFERENCE";"KLNJ1/8";"";
    "SUPPLIER";"NSK";"";
    "DESIGN";"Single row deep groove ball bearings (Inch) - Open";"";
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_traceparts_txt(path: Path) -> dict[str, str]:
    """Parse a TraceParts spec .txt file into a flat {symbol: value} dict.

    The file uses semicolons as delimiters and may have a BOM.
    Fields with empty values are omitted from the output.
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    specs: dict[str, str] = {}

    reader = csv.reader(io.StringIO(text), delimiter=";")
    for row in reader:
        if len(row) < 2:
            continue
        symbol = row[0].strip().strip('"')
        value  = row[1].strip().strip('"')
        if not symbol or not value or symbol.lower() == "symbol":
            continue
        # Normalise comma-as-decimal (European locale) in numeric fields
        specs[symbol] = value

    return specs


def scan_traceparts_dir(base_dir: Path) -> list[dict]:
    """Scan a TraceParts directory and return one entry per valid part folder.

    A valid folder has exactly one .stp file.  The .txt spec file is optional.

    Returns a list of dicts with keys:
        folder_name   — the folder basename (e.g. "180746098-19-bearing_klnj1_8_0")
        part_name     — derived from the .stp stem (e.g. "bearing_klnj1_8_0")
        stp_path      — absolute Path to the .stp file
        txt_path      — absolute Path to the .txt file, or None
        specs         — parsed spec dict (empty dict if no .txt)
        description   — human-readable description from DESIGN or PartTitle field
        supplier      — SUPPLIER field value, or ""
        reference     — REFERENCE field value, or ""
    """
    entries: list[dict] = []

    if not base_dir.exists():
        logger.warning("TraceParts directory not found: %s", base_dir)
        return entries

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        stp_files = list(folder.glob("*.stp")) + list(folder.glob("*.step"))
        if not stp_files:
            logger.debug("No .stp found in %s, skipping", folder.name)
            continue
        if len(stp_files) > 1:
            logger.warning("%s has %d .stp files — using first: %s",
                           folder.name, len(stp_files), stp_files[0].name)
        stp_path = stp_files[0]

        txt_files = list(folder.glob("*.txt"))
        txt_path  = txt_files[0] if txt_files else None

        specs: dict[str, str] = {}
        if txt_path:
            try:
                specs = parse_traceparts_txt(txt_path)
            except Exception as exc:
                logger.warning("Could not parse spec file %s: %s", txt_path.name, exc)

        description = (
            specs.get("DESIGN")
            or specs.get("TraceParts.PartTitle")
            or specs.get("DESCRIPTION")
            or stp_path.stem
        )

        entries.append({
            "folder_name": folder.name,
            "part_name":   stp_path.stem,
            "stp_path":    stp_path,
            "txt_path":    txt_path,
            "specs":       specs,
            "description": description,
            "supplier":    specs.get("SUPPLIER", ""),
            "reference":   specs.get("REFERENCE", ""),
        })

    logger.info("Found %d TraceParts entries in %s", len(entries), base_dir)
    return entries
