"""Convert all existing STEP files in data/ to USD using VLM config."""
from __future__ import annotations
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from simready.pipeline import run as pipeline_run
from simready.quality_gate import LowConfidenceError

CONFIG = PROJECT_ROOT / "temp" / "vlm_config.yaml"
STEP_DIR = PROJECT_ROOT / "data" / "step_files"
OUTPUT_DIR = PROJECT_ROOT / "output" / "simready-500"
WORKERS = 4

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("convert_all")


def _category_from_path(step_path: Path) -> str:
    """Infer output category folder from the STEP file's parent directory structure."""
    # data/step_files/<category>/freecad/<file>.step → <category>
    parts = step_path.relative_to(STEP_DIR).parts
    return parts[0] if parts else "misc_mechanical"


def _convert(step_path: Path) -> tuple[str, str, dict | None]:
    """Return (stem, status, summary). status: 'ok' | 'skipped' | 'error'."""
    category = _category_from_path(step_path)
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / (step_path.stem + ".usda")
    try:
        summary = pipeline_run(step_path, usd_path, config_path=CONFIG)
        return step_path.stem, "ok", summary
    except LowConfidenceError as e:
        return step_path.stem, "skipped", {"reason": str(e)}
    except Exception as e:
        return step_path.stem, "error", {"reason": str(e)}


def main() -> None:
    step_files = sorted(STEP_DIR.rglob("*.step")) + sorted(STEP_DIR.rglob("*.stp"))
    total = len(step_files)
    print(f"Found {total} STEP files. Converting with VLM enabled ({WORKERS} workers)...\n")

    ok = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_convert, p): p for p in step_files}
        for i, fut in enumerate(as_completed(futures), 1):
            stem, status, info = fut.result()
            if status == "ok":
                ok += 1
                q = info.get("quality_score", 0)
                mat = info.get("material_class") or "-"
                conf = info.get("material_confidence", 0)
                print(f"[{i:>3}/{total}] ✓  {stem:<45} q={q:.2f}  mat={mat:<16} conf={conf:.2f}")
            elif status == "skipped":
                skipped += 1
                print(f"[{i:>3}/{total}] ~  {stem:<45} skipped: {info['reason']}")
            else:
                errors += 1
                print(f"[{i:>3}/{total}] ✗  {stem:<45} ERROR: {info['reason']}")

    print(f"\n{'='*70}")
    print(f" Done: {ok} converted, {skipped} skipped (low confidence), {errors} errors")
    print(f" USD output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
