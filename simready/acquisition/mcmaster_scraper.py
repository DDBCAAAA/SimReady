"""mcmaster_scraper.py — McMaster-Carr product catalog builder.

McMaster-Carr requires a login for all product browsing and blocks all
automated HTTP clients (requests, headless browsers). This module therefore
works from user-provided input rather than live scraping.

Workflow
--------
1. User browses McMaster-Carr manually (logged-in browser session).
2. User collects part numbers into a plain-text list (one per line) or a CSV.
3. This module reads that list, cross-references any locally downloaded STEP
   files, and writes a structured JSON + CSV catalog.
4. Optionally, already-converted USD files are also linked in the catalog.

Input formats accepted
----------------------
Plain text  — one part number per line::

    1718K33
    7014N106
    5972K394

CSV (with optional columns)::

    part_number,name,url,notes
    1718K33,Fan Blade,,
    7014N106,Base-Mount AC Motor,https://www.mcmaster.com/7014N106/,

Output
------
    output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.json
    output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.csv
"""
from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MCMASTER_BASE = "https://www.mcmaster.com"

# Directories searched for STEP files (in order)
_STEP_SEARCH_DIRS = [
    Path("data/step_files"),
    Path("output"),
    Path("."),
]

# Directories searched for already-converted USD files
_USD_SEARCH_DIRS = [
    Path("output"),
]


# ---------------------------------------------------------------------------
# Input parsers
# ---------------------------------------------------------------------------

def _is_valid_part(s: str) -> bool:
    """Return True if s looks like a McMaster part number."""
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-]{2,14}", s.strip()))


def parse_parts_file(path: Path) -> list[dict]:
    """Parse a plain-text or CSV file of McMaster part numbers.

    Returns a list of dicts with at least 'part_number' set.
    """
    text = path.read_text(encoding="utf-8").strip()
    records: list[dict] = []
    seen: set[str] = set()

    # Detect CSV
    if "," in text.splitlines()[0] if text else "":
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            part = (row.get("part_number") or row.get("part") or "").strip().upper()
            if not part or not _is_valid_part(part) or part in seen:
                continue
            seen.add(part)
            records.append({
                "part_number": part,
                "name":     row.get("name", "").strip() or part,
                "url":      row.get("url", "").strip() or f"{MCMASTER_BASE}/{part}/",
                "notes":    row.get("notes", "").strip(),
                "has_cad":  None,   # unknown until STEP file found
                "step_file": None,
                "usd_file":  None,
                "category":  row.get("category", "").strip(),
                "specs":     {},
            })
    else:
        # Plain text — one part number per line
        for line in text.splitlines():
            part = line.strip().split()[0].upper() if line.strip() else ""
            if not part or part.startswith("#") or not _is_valid_part(part):
                continue
            if part in seen:
                continue
            seen.add(part)
            records.append({
                "part_number": part,
                "name":        part,
                "url":         f"{MCMASTER_BASE}/{part}/",
                "notes":       "",
                "has_cad":     None,
                "step_file":   None,
                "usd_file":    None,
                "category":    "",
                "specs":       {},
            })

    return records


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_step_file(part: str) -> Path | None:
    """Search standard directories for a STEP file matching part number."""
    patterns = [
        f"{part}.step", f"{part}.stp",
        f"{part}*.step", f"{part}*.stp",
        f"*{part}*.step", f"*{part}*.stp",
    ]
    for base_dir in _STEP_SEARCH_DIRS:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            matches = sorted(base_dir.rglob(pattern))
            if matches:
                return matches[0]
    return None


def _find_usd_file(part: str) -> Path | None:
    """Search standard output directories for a converted USD file."""
    for base_dir in _USD_SEARCH_DIRS:
        if not base_dir.exists():
            continue
        for pattern in [f"**/{part}/*.usd", f"**/{part}/*.usda", f"**/{part}.usd"]:
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


# ---------------------------------------------------------------------------
# API enrichment (when mTLS client cert is available)
# ---------------------------------------------------------------------------

def _enrich_via_api(records: list[dict]) -> list[dict]:
    """Call McMaster API to fill in name, category, specs, and cad_path.

    Only runs when MC_CLIENT_CERT + MC_CLIENT_KEY are configured.
    Silently skips parts that fail (network errors, subscription limits).
    """
    from simready.acquisition.mcmaster_api import McMasterCredentials, McMasterSession

    creds = McMasterCredentials()
    enriched = 0
    with McMasterSession(creds) as session:
        for rec in records:
            part = rec["part_number"]
            try:
                api_rec = session.fetch_part_record(part)
                if api_rec.get("name") and api_rec["name"] != part:
                    rec["name"] = api_rec["name"]
                if api_rec.get("category"):
                    rec["category"] = api_rec["category"]
                if api_rec.get("specs"):
                    rec["specs"].update(api_rec["specs"])
                if api_rec.get("cad_path"):
                    rec["cad_path"] = api_rec["cad_path"]
                    rec["has_cad"]  = True
                enriched += 1
                logger.info("API enriched: %s → %s", part, rec["name"][:60])
            except Exception as exc:
                logger.debug("API skip %s: %s", part, exc)
    logger.info("API enriched %d/%d records", enriched, len(records))
    return records


# ---------------------------------------------------------------------------
# Catalog enrichment
# ---------------------------------------------------------------------------

def enrich_records(records: list[dict], step_dir: Path | None = None) -> list[dict]:
    """Cross-reference STEP and USD files for each part, set has_cad."""
    for rec in records:
        part = rec["part_number"]

        # Check custom step_dir first, then standard dirs
        step = None
        if step_dir and step_dir.exists():
            for pat in [f"{part}.step", f"{part}.stp", f"*{part}*.step"]:
                hits = sorted(step_dir.glob(pat))
                if hits:
                    step = hits[0]
                    break

        if step is None:
            step = _find_step_file(part)

        rec["step_file"] = str(step) if step else None
        rec["has_cad"]   = step is not None

        usd = _find_usd_file(part)
        rec["usd_file"] = str(usd) if usd else None

        logger.debug(
            "%s | step=%s | usd=%s",
            part,
            step.name if step else "—",
            usd.name if usd else "—",
        )

    return records


# ---------------------------------------------------------------------------
# Catalog writer
# ---------------------------------------------------------------------------

def save_catalog(
    records: list[dict],
    out_dir: Path,
    tag: str,
) -> tuple[Path, Path]:
    """Write records to JSON and CSV. Returns (json_path, csv_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"mcmaster_{slug}_{ts}"

    json_path = out_dir / f"{stem}.json"
    csv_path  = out_dir / f"{stem}.csv"

    payload = {
        "tag":           tag,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "total":         len(records),
        "step_available": sum(1 for r in records if r["step_file"]),
        "usd_converted":  sum(1 for r in records if r["usd_file"]),
        "products":      records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON → %s", json_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "part_number", "name", "url",
            "has_cad", "step_file", "usd_file",
            "category", "notes", "specs",
        ])
        for r in records:
            specs_str = "; ".join(f"{k}: {v}" for k, v in r.get("specs", {}).items())
            writer.writerow([
                r["part_number"],
                r["name"],
                r["url"],
                r["has_cad"],
                r["step_file"] or "",
                r["usd_file"] or "",
                r["category"],
                r["notes"],
                specs_str,
            ])
    logger.info("CSV  → %s", csv_path)

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_catalog(
    parts_file: Path,
    tag: str = "catalog",
    step_dir: Path | None = None,
    out_dir: Path = Path("output/mcmaster_catalog"),
    use_api: bool = True,
) -> tuple[Path, Path]:
    """Build a structured McMaster product catalog from a parts list file.

    Args:
        parts_file: Path to a .txt or .csv file of McMaster part numbers.
        tag:        Label used in output filenames (e.g. "motors", "bearings").
        step_dir:   Directory to search for downloaded STEP files.
        out_dir:    Directory for JSON/CSV output.

    Returns:
        (json_path, csv_path) of the written catalog files.
    """
    if not parts_file.exists():
        raise FileNotFoundError(f"Parts file not found: {parts_file}")

    logger.info("Reading parts from: %s", parts_file)
    records = parse_parts_file(parts_file)
    logger.info("Parsed %d part numbers", len(records))

    # API enrichment — only when mTLS client cert is configured
    if use_api:
        from simready.acquisition.mcmaster_api import credentials_available, describe_missing_credentials
        if credentials_available():
            logger.info("McMaster API credentials found — enriching via API…")
            records = _enrich_via_api(records)
        else:
            logger.info(
                "McMaster API not available (mTLS client cert required).\n%s",
                describe_missing_credentials(),
            )

    # File-system enrichment — cross-reference local STEP + USD files
    records = enrich_records(records, step_dir=step_dir)

    step_count = sum(1 for r in records if r["step_file"])
    usd_count  = sum(1 for r in records if r["usd_file"])
    logger.info(
        "Enriched: %d parts | %d have STEP | %d have USD",
        len(records), step_count, usd_count,
    )

    return save_catalog(records, out_dir, tag)
