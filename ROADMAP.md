# SimReady-500: Roadmap to a Citeable Industrial Parts Dataset

## Strategic Foundation

**Target:** The definitive physics-annotated industrial parts dataset for simulated robotic assembly and bin picking.

**Why this niche, why now:** Factory automation — bin picking, kitting, assembly — is the fastest-growing segment of commercial robotics (Covariant, Dexterity, Flexiv, Realtime Robotics). Every team trains in simulation. No open dataset provides industrial parts with physics properties. NVIDIA Warehouse (753 assets) covers logistics objects (boxes, shelves, pallets) but not the parts that go on them.

**Scope:** 500 curated industrial mechanical parts across 8 categories. Every asset ships with physics annotations (density, friction, restitution, inertia tensor), semantic labels, collision meshes, and a quality score. Small enough to manually verify every asset. Large enough to be useful — NVIDIA Warehouse has 753, YCB has 77, Google Scanned Objects has 1K.

### The moat

No single feature is unique. The combination is:

1. **Physics annotations with confidence tracking** — density, friction, restitution from engineering material databases; each property carries a provenance score (0.0 = estimated, 1.0 = from source)
2. **Industrial-part focus** — the exact category bin-picking and assembly robots need, not general 3D objects
3. **Per-asset quality score** — `qualityScore`, `watertight`, `physicsComplete` in machine-readable metadata; enables programmatic filtering
4. **Multi-format** — USD + URDF + MJCF from one source; works in Isaac Lab, MuJoCo, ManiSkill, ROS without conversion
5. **Reproducible open pipeline** — `simready convert` on any STEP file; the tool is as citable as the dataset

### Honest constraints

Physics annotations come from engineering handbook defaults matched by material name classification, not from embedded CAE data in the STEP files. The pipeline tracks this via confidence scoring. The paper must be transparent about this methodology.

---

## Dataset Composition (500 assets)

| Category | Count | Primary source | Example parts |
|---|---|---|---|
| Fasteners | 120 | FreeCAD DIN/ISO library | M8 hex bolt, M6 nut, lock washer, socket cap screw |
| Gears | 60 | FreeCAD + GitHub | Spur gear, bevel gear, worm gear, rack |
| Brackets & plates | 60 | FreeCAD + NIST | L-bracket, mounting plate, angle bracket, gusset |
| Housings & enclosures | 50 | FreeCAD + GitHub | Motor housing, junction box, bearing housing |
| Connectors & fittings | 50 | FreeCAD | Pipe elbow, flange, coupling, tee fitting |
| Shafts, bearings & bushings | 50 | FreeCAD + ABC Dataset | Shoulder bolt, linear bearing, shaft collar |
| Valves & pipe components | 40 | FreeCAD | Ball valve, check valve, pipe clamp |
| Misc mechanical | 70 | Mixed | Spring, clamp, spacer, retaining ring, key, pin |

All 500 assets must pass: `physicsComplete=true`, `qualityScore >= 0.7`, `watertight=true`.

---

## Milestone 0 — Close the Quality Gap [DONE]
**Completed: March 2026**

| Item | Status |
|---|---|
| Inertia tensor (`MassAPI.diagonalInertia` + `principalAxes`) | Done |
| Center-of-mass pivot normalization | Done |
| STEP product name extraction (XDE) | Done |
| Per-asset quality score | Done |
| Tests | 79/79 passing |

---

## Milestone 1 — Curate the 500
**Target: April 2026**

### 1a. Expand material classification (7 → 25+ classes)

Highest-leverage change. Currently only 7 material classes produce `physicsComplete=true`. Industrial parts use many more materials.

Add to `_MATERIAL_CLASS_DEFAULTS`:

| Class | density (kg/m³) | friction_s | Covers |
|---|---|---|---|
| stainless | 8000 | 0.55 | Stainless steel fasteners, food-grade parts |
| cast_iron | 7200 | 0.50 | Housings, brackets, valve bodies |
| titanium | 4500 | 0.36 | Aerospace fasteners, high-strength parts |
| brass | 8500 | 0.35 | Fittings, connectors, bushings |
| bronze | 8800 | 0.40 | Bearings, bushings, worm gears |
| nylon | 1150 | 0.35 | Gears, spacers, cable ties |
| ptfe | 2200 | 0.04 | Seals, low-friction bearings |
| acetal | 1410 | 0.30 | Precision gears, cam followers |
| polycarbonate | 1200 | 0.45 | Enclosures, guards, covers |
| carbon_fiber | 1600 | 0.35 | Structural panels, drone parts |
| zinc | 7130 | 0.40 | Die-cast housings, brackets |
| ceramic | 3900 | 0.50 | Insulators, bearings |
| chrome | 7190 | 0.40 | Plated shafts, pins |

Also improve `classify_material()` to handle compound names: "stainless_steel" → stainless, "abs_plastic" → plastic_abs, "alu_6061" → aluminum.

### 1b. Manual curation pipeline

At 500 assets, manual curation is feasible and essential. Process:

```bash
# 1. Batch-acquire candidates from FreeCAD + GitHub
simready batch --source freecad --category fasteners --max-assets 200

# 2. Auto-convert and score
simready batch --convert --quality-min 0.7

# 3. Review flagged assets (watertight=false, low confidence)
simready catalog --filter "quality_score < 0.7" --format table

# 4. Manually assign material class where auto-classification fails
simready tag asset_id --material steel --category fastener:bolt
```

Every asset in the final dataset must be human-verified at least once.

### 1c. Batch pipeline

- `simready batch` — async acquire + convert + score in one command
- Per-asset error isolation (one failure doesn't kill the batch)
- Progress reporting: processed / passed / failed / skipped
- `--workers N` for parallel conversion

### 1d. SQLite catalog

Replace `data/catalog.json` with `data/catalog.db`:
- Indexed by: category, material class, quality score, physics completeness
- `simready catalog --query "category=fastener AND material_class=steel"` < 1s
- JSON export: `simready catalog --format json`

**Exit criterion:** 500 assets in catalog, all with `physicsComplete=true` and `qualityScore >= 0.7`. Category distribution matches the table above (within +/-20%).

---

## Milestone 2 — Multi-Format Export
**Target: May 2026**

Every asset ships in every format a robotics lab might need. At 500 assets, we can afford to verify every export.

```
simready-500/
  fasteners/
    bolt_m8_hex/
      ├── bolt_m8_hex.usd         (Omniverse / Isaac Lab)
      ├── bolt_m8_hex.urdf        (ROS / MoveIt)
      ├── bolt_m8_hex.xml         (MuJoCo / ManiSkill)
      ├── meshes/
      │   ├── visual.obj          (shared visual mesh)
      │   └── collision.obj       (convex hull)
      ├── renders/
      │   ├── view_000.png        (8 × RGB)
      │   ├── view_000_depth.exr  (8 × depth)
      │   └── view_000_normal.png (8 × normals)
      └── metadata.json
  gears/
    spur_gear_m2/
      └── ...
```

### Work items

- `simready/export/urdf_writer.py` — body hierarchy, inertial (mass, inertia tensor from USD), collision + visual meshes as OBJ references
- `simready/export/mjcf_writer.py` — geom, body, inertial, friction/condim from physics material
- `simready/render/headless.py` — BlenderProc headless: 8 azimuth views at 30° elevation, 512×512, neutral studio HDRI
- `metadata.json` schema per asset:

```json
{
  "name": "bolt_m8_hex",
  "category": "fastener:bolt",
  "material_class": "steel",
  "physics": {
    "density_kg_m3": 7850.0,
    "friction_static": 0.55,
    "friction_dynamic": 0.42,
    "restitution": 0.3,
    "inertia_principal": [0.001, 0.001, 0.0005],
    "mass_kg": 0.032
  },
  "quality": {
    "score": 0.92,
    "watertight": true,
    "physics_complete": true,
    "material_confidence": 0.25
  },
  "source": {
    "url": "https://github.com/FreeCAD/FreeCAD-library/...",
    "license": "LGPL-2.1",
    "pipeline_version": "0.3.0"
  },
  "geometry": {
    "vertices": 2841,
    "faces": 5678,
    "lod_levels": 3,
    "bounding_box_m": [0.013, 0.013, 0.025]
  }
}
```

**Exit criterion:** All 500 assets have USD + URDF + MJCF + metadata.json + renders. Spot-check: 20 random assets load correctly in Isaac Lab, MuJoCo, and RViz.

---

## Milestone 3 — Publication
**Target: June 2026**

### 3a. HuggingFace dataset

```python
from datasets import load_dataset
ds = load_dataset("simready/industrial-parts-500", split="train")
asset = ds[0]  # renders, metadata, USD/URDF/MJCF blobs
```

- Dataset card: category breakdown, physics property distributions, quality histograms, license composition
- `simready push --hf-repo simready/industrial-parts-500`

### 3b. Simulation benchmarks

Reproducible scripts in `benchmarks/`. Two experiments, three conditions each:

**Condition A:** SimReady assets with full physics (curated density, friction, restitution, inertia)
**Condition B:** Same meshes, uniform default physics (density=1000, friction=0.5, restitution=0.3)
**Condition C:** Same meshes, no physics (density=1, friction=0, restitution=0)

**Experiment 1 — Drop-test stability (Isaac Sim):**
Drop all 500 objects onto a flat surface. Measure penetration depth, settling time, rest-state stability. Expected result: Condition A settles correctly; C falls through or bounces infinitely.

**Experiment 2 — Bin-picking grasp success (Isaac Lab):**
Franka parallel-jaw grasping from a bin of 50 randomly selected parts. 1000 trials per condition. Metric: grasp success rate + post-grasp slip rate. Expected result: Condition A has higher grasp success because friction is realistic; C has excessive slip.

Both experiments scripted, single-GPU reproducible: `python benchmarks/run_drop_test.py`, `python benchmarks/run_bin_pick.py`.

### 3c. Paper

**Title:** *SimReady-500: Physics-Annotated Industrial Parts for Simulated Robotic Assembly*

**Thesis:** Physics annotations on industrial parts — density, friction, inertia from engineering material databases — measurably improve simulation fidelity for bin-picking and assembly tasks. We release a curated, multi-format dataset of 500 parts with transparent quality scoring.

**Structure:**
1. **Introduction** — sim2real gap in industrial manipulation; existing datasets lack physics
2. **Pipeline** — STEP → USD conversion, material classification, quality scoring (reproducible, open-source)
3. **Dataset** — 500 parts, 8 categories, statistics, category/material distributions
4. **Benchmarks** — drop-test stability + bin-picking grasp success across three physics conditions
5. **Comparison** — SimReady vs. ShapeNet vs. Objaverse vs. ABC Dataset vs. NVIDIA Warehouse (feature matrix)
6. **Limitations** — material properties from handbook defaults, keyword classification coverage, no articulated assemblies

**Target venues:** CoRL 2026, RA-L rolling, ICRA 2027. arXiv preprint immediately.

**Exit criterion:** arXiv live, HuggingFace published, benchmarks reproducible.

---

## Milestone 4 — Ecosystem Integration
**Target: July+ 2026, ongoing**

- **`simready.load()` API**: `simready.load("fastener:bolt", format="mjcf", min_quality=0.8)`
- **Isaac Lab loader**: PR to `isaac-lab/isaac-lab` with `SimReadyAssetCfg`
- **ManiSkill integration**: SimReady as built-in asset source
- **Procedural augmentation**: texture/scale/material variation from single master — multiply 500 → 5000 training variants
- **Assembly hierarchy**: STEP assembly tree → nested USD Xforms; kinematic joints → `UsdPhysics.RevoluteJoint`
- **v2 expansion**: grow from 500 → 2000 by adding categories (electrical components, pneumatics, sensors)

---

## Execution Timeline

```
Mar 2026  ████ M0: Quality gap closed (DONE)
Apr 2026  ████████ M1: 500 curated industrial parts, material expansion, batch pipeline
May 2026  ████████████ M2: Multi-format export (URDF, MJCF, renders, metadata.json)
Jun 2026  ████████████████ M3: Paper + HuggingFace + benchmarks
Jul+      ████████████████ M4: Isaac Lab PR, ManiSkill, simready.load(), augmentation
```

---

## Why This Gets Cited

1. **The only physics-annotated industrial parts dataset** — density, friction, inertia on every asset, with provenance tracking
2. **Purpose-built for bin picking and assembly** — the exact objects factory-automation labs need, not general 3D meshes
3. **Multi-format from one source** — USD + URDF + MJCF; works in Isaac Lab, MuJoCo, ManiSkill, ROS without conversion
4. **500 verified assets > 50K unverified** — every asset human-reviewed, 100% physics complete, 100% watertight
5. **Benchmark proof** — drop-test and bin-picking experiments quantify the value of physics annotations
6. **Reproducible pipeline** — labs can run `simready convert` on their own STEP files; tool + dataset both citable

Comparable datasets by scale: NVIDIA Warehouse (753), YCB (77), Google Scanned Objects (1K), ContactDB (50). SimReady-500 fits squarely in the range that gets cited.
