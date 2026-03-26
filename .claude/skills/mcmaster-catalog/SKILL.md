---
name: mcmaster-catalog
description: Build a structured McMaster-Carr product catalog (JSON + CSV) from a manually collected parts list. Records part number, name, URL, STEP file status, USD conversion status, category, and specs. McMaster requires login and blocks all automated clients, so part numbers must be collected manually.
---

# McMaster-Carr Product Catalog Skill

Use this skill when the user asks to catalog, organize, or track McMaster-Carr
products — especially for managing downloaded STEP files and USD conversions.

## Why manual input is required

McMaster-Carr requires a logged-in account for all product browsing and
actively blocks all automated HTTP clients (requests, headless browsers,
Playwright, Selenium). This is confirmed and cannot be bypassed without
violating their Terms of Service.

## Workflow

```
Step 1 — Browse www.mcmaster.com (logged in) and collect part numbers
Step 2 — Save them to a .txt or .csv file
Step 3 — python mcmaster_catalog.py parts.txt --tag <label>
Step 4 — Download STEP files manually from each product URL (browser)
Step 5 — python mcmaster_to_usd.py --part <PART> --file ~/Downloads/<FILE>.step
```

## Step 1 — Creating the parts file

**Plain text** (simplest — one part number per line):
```
1718K33
7014N106
5972K394
```

**CSV** (richer — with optional name, url, category, notes):
```csv
part_number,name,url,category,notes
7014N106,Base-Mount AC Motor,https://www.mcmaster.com/7014N106/,Motors,60Hz
1718K33,Fan Blade,,Fans,
5972K394,Ball Bearing,,Bearings,10mm bore
```

## Step 3 — Running the catalog builder

```bash
python mcmaster_catalog.py parts.txt
python mcmaster_catalog.py parts.txt  --tag motors     --step-dir ~/Downloads
python mcmaster_catalog.py parts.csv  --tag bearings   --out output/bearings
python mcmaster_catalog.py parts.txt  --verbose
```

### Flags

| Flag | Meaning |
|---|---|
| `--tag LABEL` | Label for output filenames (default: `catalog`) |
| `--step-dir DIR` | Directory with downloaded STEP files (auto-matched by part number) |
| `--out DIR` | Output directory (default: `output/mcmaster_catalog/`) |
| `--verbose` | Debug logging |

## Output files

```
output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.json
output/mcmaster_catalog/mcmaster_<tag>_<timestamp>.csv
```

### JSON structure

```json
{
  "tag": "motors",
  "generated_at": "2026-03-25T10:00:00Z",
  "total": 5,
  "step_available": 3,
  "usd_converted": 1,
  "products": [
    {
      "part_number": "7014N106",
      "name": "Base-Mount AC Motor",
      "url": "https://www.mcmaster.com/7014N106/",
      "has_cad": true,
      "step_file": "output/7014N106/7014N106.step",
      "usd_file": "output/7014N106/7014N106.usd",
      "category": "Motors",
      "notes": "60Hz",
      "specs": {}
    }
  ]
}
```

### CSV columns

`part_number`, `name`, `url`, `has_cad`, `step_file`, `usd_file`, `category`, `notes`, `specs`

## Full pipeline example

```bash
# 1. After browsing McMaster manually, create parts.txt:
echo "7014N106" > parts.txt
echo "1718K33"  >> parts.txt

# 2. Build the catalog (cross-references any downloaded STEP files)
python mcmaster_catalog.py parts.txt --tag my_parts --step-dir ~/Downloads

# 3. Convert each downloaded STEP to USD
python mcmaster_to_usd.py --part 7014N106 --file ~/Downloads/7014N106_Motor.STEP
python mcmaster_to_usd.py --part 1718K33  --file ~/Downloads/1718K33.step

# 4. Re-run catalog to update USD status
python mcmaster_catalog.py parts.txt --tag my_parts
```
