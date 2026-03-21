#!/usr/bin/env bash
# run_batch_all.sh — run the SimReady-500 batch pipeline across all categories.
# Usage: bash scripts/run_batch_all.sh
set -euo pipefail

W=4          # parallel workers
QM=0.5       # quality threshold

echo "============================================================"
echo " SimReady-500 Batch Pipeline"
echo "============================================================"

run() {
    local label="$1"; shift
    echo ""
    echo ">>> [$label] $*"
    python -m simready.cli batch "$@" --workers "$W" --quality-min "$QM" --source freecad
}

# --- Fasteners (~100) --------------------------------------------------------
run "fasteners/bolt"      --category bolt      --material steel         --max-assets 50
run "fasteners/screw"     --category screw     --material steel         --max-assets 30
run "fasteners/nut"       --category nut       --material steel         --max-assets 20

# --- Gears (~80) --------------------------------------------------------------
run "gears/gear"          --category gear      --material steel         --max-assets 50
run "gears/sprocket"      --category sprocket  --material steel         --max-assets 15
run "gears/pulley"        --category pulley    --material steel         --max-assets 15

# --- Housings & Enclosures (~60) ---------------------------------------------
run "housings/housing"    --category housing   --material cast_aluminum --max-assets 40
run "housings/enclosure"  --category enclosure --material cast_aluminum --max-assets 20

# --- Brackets & Plates (~30) -------------------------------------------------
run "brackets/bracket"    --category bracket   --material aluminum      --max-assets 20
run "brackets/plate"      --category plate     --material steel         --max-assets 10

# --- Connectors & Fittings (~35) --------------------------------------------
run "connectors/fitting"  --category fitting   --material steel         --max-assets 20
run "connectors/coupling" --category coupling  --material steel         --max-assets 15

# --- Shafts & Bearings (~30) -------------------------------------------------
run "shafts/shaft"        --category shaft     --material steel         --max-assets 20
run "shafts/bearing"      --category bearing   --material steel         --max-assets 10

# --- Valves & Pipe (~20) -----------------------------------------------------
run "valves/valve"        --category valve     --material steel         --max-assets 10
run "valves/pipe"         --category pipe      --material steel         --max-assets 10

# --- Misc Mechanical (~45) ---------------------------------------------------
run "misc/spring"         --category spring    --material steel         --max-assets 20
run "misc/pin"            --category pin       --material steel         --max-assets 15
run "misc/cam"            --category cam       --material steel         --max-assets 10

echo ""
echo "============================================================"
echo " All batches complete. Run 'simready catalog' to review."
echo "============================================================"
