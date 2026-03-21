#!/usr/bin/env bash
# init_dirs.sh — create the SimReady-500 folder skeleton.
# Run once after a fresh clone. Safe to re-run (idempotent).
set -euo pipefail

CATEGORIES=(
  fasteners
  gears
  brackets_plates
  housings_enclosures
  connectors_fittings
  shafts_bearings
  valves_pipe
  misc_mechanical
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Initialising SimReady directory structure at $ROOT"

for cat in "${CATEGORIES[@]}"; do
  # Input: downloaded STEP files, organised by category
  mkdir -p "$ROOT/data/step_files/$cat"
  touch    "$ROOT/data/step_files/$cat/.gitkeep"

  # Output: converted assets per category
  mkdir -p "$ROOT/output/simready-500/$cat"
  touch    "$ROOT/output/simready-500/$cat/.gitkeep"
done

echo ""
echo "data/step_files/"
ls "$ROOT/data/step_files/"

echo ""
echo "output/simready-500/"
ls "$ROOT/output/simready-500/"

echo ""
echo "Done. Run 'simready batch --category fasteners --source freecad' to start acquiring assets."
