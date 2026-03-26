"""CLI entry point for the SimReady pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from simready import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

# All default paths are absolute, anchored to the project root, so the CLI
# works correctly regardless of the working directory it is invoked from.
_DEFAULT_STEP_DIR    = PROJECT_ROOT / "data" / "step_files"
_DEFAULT_CATALOG     = PROJECT_ROOT / "data" / "catalog.json"
_DEFAULT_CATALOG_DB  = PROJECT_ROOT / "data" / "catalog.db"
_DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "output" / "simready-500"


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("convert", help="Convert a CAD file to Sim-Ready USD")
    p.add_argument("--input", "-i", type=Path, required=True, help="Source CAD file (.step/.stp)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help=f"Output USD file path (default: {_DEFAULT_OUTPUT_DIR}/<input_stem>.usd)")
    p.add_argument("--config", "-c", type=Path, default=None, help="Pipeline config YAML")


def _add_acquire_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("acquire", help="Search and download open-source STEP files")
    p.add_argument("query", nargs="?", default=None, help="Search query (e.g. 'gear', 'bracket', 'automotive')")
    p.add_argument("--dest", "-d", type=Path, default=_DEFAULT_STEP_DIR, help="Download directory")
    p.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Catalog JSON path")
    p.add_argument("--max-results", "-n", type=int, default=10, help="Max results per source")
    p.add_argument("--sources", "-s", nargs="*", default=None, help="Sources to search (default: all)")
    p.add_argument("--search-only", action="store_true", help="Search without downloading")
    p.add_argument("--list-sources", action="store_true", help="List available sources and exit")


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Acquire STEP files and convert them to USD in one step")
    p.add_argument("query", help="Search query (e.g. 'gear', 'bracket')")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT_DIR, help="USD output directory")
    p.add_argument("--dest", "-d", type=Path, default=_DEFAULT_STEP_DIR, help="STEP download directory")
    p.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Catalog JSON path")
    p.add_argument("--max-results", "-n", type=int, default=5, help="Max STEP files to acquire per source")
    p.add_argument("--sources", "-s", nargs="*", default=None, help="Sources to search (default: all)")
    p.add_argument("--config", "-c", type=Path, default=None, help="Pipeline config YAML")


def _add_catalog_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("catalog", help="View and query the asset catalog")
    p.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Catalog JSON export path")
    p.add_argument("--db", type=Path, default=_DEFAULT_CATALOG_DB, help="SQLite catalog path")
    p.add_argument("--query", "-q", default=None,
                   help="SQL WHERE filter, e.g. \"category='fastener' AND physics_complete=1\"")
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.add_argument("--downloaded-only", action="store_true", help="Only show downloaded assets")


def _add_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("batch", help="Batch acquire, convert, and score assets")
    p.add_argument("--source", choices=["freecad", "github", "abc", "nist"], default=None,
                   help="Acquisition source (default: all)")
    p.add_argument("--category", default=None, help="Search query / category keyword")
    p.add_argument("--max-assets", "-n", type=int, default=10, help="Max assets to process")
    p.add_argument("--quality-min", type=float, default=0.5,
                   help="Minimum quality score to pass (default: 0.5)")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel conversion workers")
    p.add_argument("--output-dir", "-o", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--dest", "-d", type=Path, default=_DEFAULT_STEP_DIR)
    p.add_argument("--db", type=Path, default=_DEFAULT_CATALOG_DB)
    p.add_argument("--config", "-c", type=Path, default=None)
    p.add_argument("--material", "-m", default=None,
                   help="Default material class for all assets (e.g. steel, aluminum, nylon)")


def _add_partnet_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "partnet",
        help="Download and convert PartNet-Mobility objects to articulated USD",
    )
    p.add_argument(
        "object_ids", nargs="+",
        help="PartNet-Mobility object IDs (e.g. 101516 102379)",
    )
    p.add_argument(
        "--data-dir", "-d", type=Path,
        default=PROJECT_ROOT / "data" / "partnet",
        help="Download directory (default: data/partnet)",
    )
    p.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "output" / "partnet",
        help="USD output directory (default: output/partnet)",
    )
    p.add_argument("--category", default=None,
                   help="Object category hint for material inference (e.g. StorageFurniture)")
    p.add_argument("--material", "-m", default=None,
                   help="Force material class for all links (e.g. wood, steel)")
    p.add_argument("--vlm", action="store_true",
                   help="Enable VLM material inference via Claude API")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace API token (only needed for gated repos)")
    p.add_argument("--config", "-c", type=Path, default=None)


def _add_traceparts_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "traceparts",
        help="Batch-convert TraceParts CAD folders (.stp + .txt) to USD",
    )
    p.add_argument(
        "--dir", "-d", type=Path,
        default=PROJECT_ROOT / "data" / "TraceParts",
        help="Root TraceParts directory (default: data/TraceParts)",
    )
    p.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "output" / "traceparts",
        help="USD output directory (default: output/traceparts)",
    )
    p.add_argument("--config", "-c", type=Path, default=None, help="Pipeline config YAML")
    p.add_argument(
        "--material", "-m", default=None,
        help="Force material class for all parts (e.g. steel, nylon)",
    )


def _add_sldprt_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("sldprt2step", help="Convert a SolidWorks SLDPRT file to STEP")
    p.add_argument("input", type=Path, help="Input .sldprt file")
    p.add_argument("output", type=Path, nargs="?", default=None,
                   help="Output .step file (default: same name/location as input)")
    p.add_argument("--client-id",     default=None, help="Autodesk APS Client ID (or set AUTODESK_CLIENT_ID)")
    p.add_argument("--client-secret", default=None, help="Autodesk APS Client Secret (or set AUTODESK_CLIENT_SECRET)")


def _add_tag_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tag", help="Manually curate an asset in the catalog")
    p.add_argument("asset_id", type=int, help="Catalog asset ID (from simready catalog)")
    p.add_argument("--material", default=None, help="Force material class (e.g. steel, nylon)")
    p.add_argument("--category", default=None, help="Force category (e.g. fastener:bolt)")
    p.add_argument("--db", type=Path, default=_DEFAULT_CATALOG_DB)
    p.add_argument("--config", "-c", type=Path, default=None)


def _run_convert(args: argparse.Namespace) -> None:
    from simready.pipeline import run

    output = args.output
    if output is None:
        output = _DEFAULT_OUTPUT_DIR / (args.input.stem + ".usd")
    run(args.input, output, getattr(args, "config", None))


def _run_acquire(args: argparse.Namespace) -> None:
    # Import sources to trigger registration
    import simready.acquisition.github_source  # noqa: F401
    import simready.acquisition.abc_dataset    # noqa: F401
    import simready.acquisition.nist_source    # noqa: F401
    import simready.acquisition.freecad_source # noqa: F401
    from simready.acquisition.sources import list_sources
    from simready.acquisition.agent import acquire

    if args.list_sources:
        print("Available sources:")
        for name in list_sources():
            print(f"  - {name}")
        return

    if not args.query:
        print("Error: query is required (unless using --list-sources)")
        sys.exit(1)

    assets = asyncio.run(
        acquire(
            query=args.query,
            dest_dir=args.dest,
            catalog_path=args.catalog,
            max_per_source=args.max_results,
            sources=args.sources,
            download=not args.search_only,
        )
    )

    if not assets:
        print("No assets found.")
        return

    print(f"\n{'Downloaded' if not args.search_only else 'Found'} {len(assets)} assets:\n")
    for asset in assets:
        status = f"[{asset.local_path}]" if asset.local_path else "[not downloaded]"
        license_info = f" ({asset.license})" if asset.license else ""
        print(f"  {asset.name}{license_info}")
        print(f"    source: {asset.source}")
        print(f"    {status}")
        if asset.description:
            print(f"    {asset.description[:80]}")
        print()


def _run_pipeline(args: argparse.Namespace) -> None:
    import simready.acquisition.github_source  # noqa: F401
    import simready.acquisition.abc_dataset    # noqa: F401
    import simready.acquisition.nist_source    # noqa: F401
    import simready.acquisition.freecad_source # noqa: F401
    from simready.acquisition.agent import acquire, convert_acquired_assets, save_catalog

    assets = asyncio.run(
        acquire(
            query=args.query,
            dest_dir=args.dest,
            catalog_path=args.catalog,
            max_per_source=args.max_results,
            sources=args.sources,
        )
    )

    if not assets:
        print("No assets downloaded.")
        return

    results = convert_acquired_assets(assets, args.output, getattr(args, "config", None))

    # Persist updated usd_path values to catalog
    save_catalog(assets, args.catalog)

    if not results:
        print("No files converted.")
        return

    print(f"\nConverted {len(results)} STEP file(s) to USD:\n")
    for step_path, usd_path in results:
        print(f"  {step_path.name} → {usd_path}")
    print()


def _run_catalog(args: argparse.Namespace) -> None:
    from simready.catalog.db import open_db, query_assets, export_json

    db_path = args.db
    # Fall back to JSON view if DB doesn't exist yet
    if not db_path.exists():
        from simready.acquisition.agent import load_catalog
        assets = load_catalog(args.catalog)
        if not assets:
            print("Catalog is empty.")
            return
        if args.downloaded_only:
            assets = [a for a in assets if a.local_path]
        print(f"Catalog (JSON): {len(assets)} assets\n")
        for asset in assets:
            downloaded = "Y" if asset.local_path else "N"
            print(f"  [{downloaded}] {asset.name} ({asset.source})")
            if asset.local_path:
                print(f"      → {asset.local_path}")
        print()
        return

    conn = open_db(db_path)
    filter_expr = args.query
    if args.downloaded_only:
        filter_expr = (f"({filter_expr}) AND local_path IS NOT NULL"
                       if filter_expr else "local_path IS NOT NULL")

    if args.format == "json":
        export_json(conn, args.catalog, filter_expr)
        print(f"Exported to {args.catalog}")
        conn.close()
        return

    entries = query_assets(conn, filter_expr)
    conn.close()

    if not entries:
        print("No assets match the query.")
        return

    print(f"Catalog: {len(entries)} asset(s)\n")
    print(f"  {'ID':>4}  {'Name':<38} {'Score':>6}  {'Phys':>5}  {'Wet':>4}  {'Category':<22}  {'MatClass'}")
    print("  " + "-" * 98)
    for e in entries:
        print(
            f"  {e.id or 0:>4}  {e.name:<38} {e.quality_score or 0.0:>6.2f}  "
            f"{'yes' if e.physics_complete else 'no':>5}  "
            f"{'yes' if e.watertight else 'no':>4}  "
            f"{(e.category or '-'):<22}  {e.material_class or '-'}"
        )
    print()


def _run_batch(args: argparse.Namespace) -> None:
    from simready.batch import run_batch, print_batch_summary

    results = run_batch(
        source=args.source,
        category=args.category,
        max_assets=args.max_assets,
        quality_min=args.quality_min,
        workers=args.workers,
        output_dir=args.output_dir,
        db_path=args.db,
        config_path=getattr(args, "config", None),
        dest_dir=args.dest,
        material_default=getattr(args, "material", None),
    )
    print_batch_summary(results)


def _run_partnet(args: argparse.Namespace) -> None:
    from simready.acquisition.partnet_adapter import convert_partnet_batch

    results = convert_partnet_batch(
        object_ids      = args.object_ids,
        data_dir        = args.data_dir,
        output_dir      = args.output,
        category_hint   = getattr(args, "category", None),
        forced_material = getattr(args, "material", None),
        enable_vlm      = getattr(args, "vlm", False),
        hf_token        = getattr(args, "hf_token", None),
    )
    passed = sum(1 for r in results if r.get("success"))
    print(f"\nPartNet batch: {passed}/{len(results)} converted\n")
    for r in results:
        if r.get("success"):
            print(f"  [OK]  {r['object_id']:>8}  quality={r.get('quality_score', 0):.2f}"
                  f"  mat={r.get('material_class', '-')}")
            print(f"          → {r['usd_path']}")
        else:
            print(f"  [ERR] {r['object_id']:>8}  {r.get('error', '')}")
    print()


def _run_traceparts(args: argparse.Namespace) -> None:
    from simready.acquisition.traceparts_reader import scan_traceparts_dir
    from simready.pipeline import run as pipeline_run

    entries = scan_traceparts_dir(args.dir)
    if not entries:
        print(f"No TraceParts parts found in {args.dir}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    overrides = {"*": args.material} if getattr(args, "material", None) else None

    passed, failed = 0, 0
    for entry in entries:
        part_name = entry["part_name"]
        stp_path  = entry["stp_path"]
        usd_path  = args.output / part_name / f"{part_name}.usd"

        # Build asset_metadata from TraceParts spec fields
        meta: dict = {}
        if entry["description"]:
            meta["description"] = entry["description"]
        if entry["supplier"]:
            meta["supplier"] = entry["supplier"]
        if entry["reference"]:
            meta["reference"] = entry["reference"]
        for k, v in entry["specs"].items():
            if k not in ("DESIGN", "SUPPLIER", "REFERENCE") and v:
                meta[k] = v

        print(f"\n[traceparts] {part_name}")
        if entry["description"]:
            print(f"  {entry['description']}")
        try:
            summary = pipeline_run(
                stp_path, usd_path,
                config_path=getattr(args, "config", None),
                material_overrides=overrides,
                asset_metadata=meta or None,
                disable_confidence_gate=True,
            )
            print(
                f"  → {usd_path}  "
                f"quality={summary['quality_score']:.2f}  "
                f"mat={summary['material_class']}"
            )
            passed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\nTraceParts batch complete: {passed} converted, {failed} failed.\n")


def _run_sldprt(args: argparse.Namespace) -> None:
    from simready.ingestion.sldprt_converter import convert_sldprt

    src = args.input
    dst = args.output or src.with_suffix(".step")
    step_path = convert_sldprt(
        src, dst,
        autodesk_client_id=args.client_id,
        autodesk_client_secret=args.client_secret,
    )
    print(f"Converted: {src} → {step_path}  ({step_path.stat().st_size // 1024} KB)")


def _run_tag(args: argparse.Namespace) -> None:
    from simready.catalog.db import open_db, get_asset, update_asset_fields

    conn = open_db(args.db)
    entry = get_asset(conn, args.asset_id)
    if entry is None:
        print(f"Error: asset {args.asset_id} not found in catalog.")
        sys.exit(1)

    updates: dict = {}
    if args.material:
        updates["material_class"] = args.material
    if args.category:
        updates["category"] = args.category

    if updates:
        update_asset_fields(conn, args.asset_id, **updates)
        print(f"Catalog updated for asset {args.asset_id}: {updates}")

    # Re-run pipeline if STEP file is available
    if entry.local_path and Path(entry.local_path).exists() and entry.usd_path:
        from simready.pipeline import run as pipeline_run
        overrides = {"*": args.material} if args.material else None
        try:
            summary = pipeline_run(
                Path(entry.local_path),
                Path(entry.usd_path),
                config_path=getattr(args, "config", None),
                material_overrides=overrides,
            )
            update_asset_fields(
                conn, args.asset_id,
                quality_score=summary["quality_score"],
                physics_complete=int(summary["physics_complete"]),
                watertight=int(summary["watertight"]),
                material_confidence=summary["material_confidence"],
                face_count=summary["face_count"],
            )
            print(f"Re-exported USD with quality_score={summary['quality_score']:.2f}")
        except Exception as exc:
            print(f"Warning: USD re-export failed ({exc}). Catalog metadata updated only.")
    else:
        if entry.local_path and not Path(entry.local_path).exists():
            print("STEP file not found — catalog metadata updated, USD not re-exported.")

    conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="simready",
        description="SimReady: Automated Sim-Ready OpenUSD asset pipeline",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_parser(subparsers)
    _add_acquire_parser(subparsers)
    _add_catalog_parser(subparsers)
    _add_run_parser(subparsers)
    _add_batch_parser(subparsers)
    _add_tag_parser(subparsers)
    _add_sldprt_parser(subparsers)
    _add_traceparts_parser(subparsers)
    _add_partnet_parser(subparsers)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        if args.command == "convert":
            _run_convert(args)
        elif args.command == "acquire":
            _run_acquire(args)
        elif args.command == "catalog":
            _run_catalog(args)
        elif args.command == "run":
            _run_pipeline(args)
        elif args.command == "batch":
            _run_batch(args)
        elif args.command == "tag":
            _run_tag(args)
        elif args.command == "sldprt2step":
            _run_sldprt(args)
        elif args.command == "traceparts":
            _run_traceparts(args)
        elif args.command == "partnet":
            _run_partnet(args)
    except Exception as e:
        logging.getLogger("simready").error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
