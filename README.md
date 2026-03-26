# SimReady

**AIGC-to-SimReady pipeline — turn raw Text-to-3D meshes into physics-annotated OpenUSD assets.**

SimReady takes the noisy, unscaled, unposed `.glb`/`.obj` outputs from generative 3D models (TRELLIS, Tripo, Meshy, Rodin) and refines them into simulation-ready USD assets with correct scale, canonical pose, collision hulls, mass/inertia, and PBR materials. Pure Python. `pip install`. No GPU runtime, no simulation engine required.

---

## The Problem

AIGC 3D models are visually plausible but physically useless:

| What's wrong | Why it matters |
|---|---|
| **Arbitrary scale** — a "chair" could be 0.001 or 100 units tall | Physics engines compute forces from mass and distance; wrong scale = wrong physics |
| **Random pose** — no consistent up-axis or resting orientation | Objects spawn sideways or upside-down in simulation |
| **Dirty geometry** — floating artifacts, degenerate faces, non-manifold edges | Collision solvers explode on non-convex or broken meshes |
| **No physics** — zero mass, zero inertia, no collision geometry | Objects are ghosts — they can't be grasped, stacked, or dropped |
| **No materials** — vertex colors at best, no PBR, no density | Friction and restitution are undefined; grasping and contact are meaningless |

SimReady fixes all of these automatically.

---

## How It Works

Two-stage architecture: **AIGC Mesh Healer** (frontend) → **Physics Core** (backend).

```
Raw AIGC mesh (.glb / .obj / .fbx)
        │
        ▼  AIGC Mesh Healer (FRONTEND)
        │  ─ Keep largest connected component (drop floating artifacts)
        │  ─ Remove degenerate faces + unreferenced vertices
        │  ─ Merge duplicate vertices, fix normals, fill holes
        │  ─ PCA pose alignment → Z-up, bottom on ground plane
        │  ─ Scale normalization → real-world meters (lookup dictionary)
        │
        ▼  Physics Core (BACKEND)
        │  ─ CoACD convex decomposition → collision hull set
        │  ─ Analytical mass/inertia via trimesh (density × volume)
        │  ─ VLM material inference (Claude API) → PBR + physics density
        │  ─ USD assembly via usd-core
        │
        ▼
SimReady .usda — physics-annotated, engine-agnostic
        Loadable in: Isaac Lab, MuJoCo (MJCF), ROS (URDF)
```

---

## Quick Start

```bash
# Install
git clone https://github.com/DDBCAAAA/SimReady.git
cd SimReady
pip install -e ".[dev]"

# Heal and convert a single AIGC mesh (no API key required)
simready heal \
  --input  data/aigc_raw/microwave.glb \
  --output output/microwave.usda \
  --object-type microwave

# With VLM material inference (requires ANTHROPIC_API_KEY in .env)
simready heal \
  --input  data/aigc_raw/office_chair.glb \
  --output output/office_chair.usda \
  --object-type office_chair \
  --vlm

# Batch-heal a directory of AIGC meshes
simready heal-batch \
  --input-dir  ./data/aigc_raw/ \
  --output-dir ./output/ \
  --manifest   prompts.csv \
  --vlm

# Generate raw meshes from text prompts (TRELLIS)
simready generate \
  --prompts prompts.txt \
  --output-dir ./data/aigc_raw/ \
  --model trellis

# VLM quality check on a generated asset
simready qc --input output/microwave.usda --render
```

---

## Output Structure

```
/Root                              Z-up · meters · UsdGeom stage
  /Root/Materials/
    plastic_abs                    OmniPBR MDL shader
                                   + UsdPhysics.MaterialAPI (friction, restitution)
  /Root/<ObjectName>               UsdGeom.Xform
    RigidBodyAPI
    MassAPI                        analytical mass, CoM, inertia tensor
    MaterialBindingAPI
    /Collision_0 … N               CoACD convex hull(s) — invisible
      CollisionAPI
    customData:
      simready:qualityScore        0.87
      simready:watertight          true
      simready:physicsComplete     true
      simready:materialConfidence  0.85
      aigc:model                   "trellis-v1"
      aigc:prompt                  "microwave oven, household appliance"
      aigc:scale_source            "dimension_dict"
```

---

## Quality Score

Each asset receives a composite quality score in USD `customData`:

| Component | Weight | Full credit when… |
|---|---|---|
| Watertight mesh | 30% | Closed manifold after healing |
| Physics complete | 40% | Density + friction + restitution + inertia all present |
| Material confidence | 15% | VLM confidence ≥ 0.7 |
| Face density | 15% | ≥ 1,000 faces |

---

## Material Classification

AIGC meshes have zero embedded material data. Resolution order:

1. **VLM inference** (primary) — Claude API classifies material from object type + rendered preview + mesh geometry. Returns material class, PBR properties, and physics density. Confidence: 70–95%.
2. **Object-type hint** — Maps object type to default material: `"microwave" → steel`, `"chair" → plastic_abs`, `"mug" → ceramic`.
3. **Fallback** — Neutral OmniPBR grey plastic (density=1050 kg/m³, confidence=0%).

25+ material classes with physics properties: `steel`, `stainless`, `aluminum`, `brass`, `bronze`, `copper`, `titanium`, `cast_iron`, `plastic_abs`, `nylon`, `rubber`, `glass`, `ceramic`, `chrome`, `wood`, and more.

---

## AIGC Mesh Healer — What It Fixes

### Geometry Healing
1. **Largest connected component** — Drops floating artifacts, stray vertices, internal geometry. Keeps only the main object.
2. **Degenerate face removal** — Faces with near-zero area cause NaN in physics solvers.
3. **Vertex cleanup** — Removes unreferenced vertices, merges duplicates within tolerance.
4. **Normal repair** — Fixes inverted normals and inconsistent winding.
5. **Hole filling** — Attempts watertight repair. Non-watertight meshes fall back to bounding-box volume approximation for mass.

### Scale Normalization
AIGC models have no concept of real-world size. The healer maps meshes to physically correct dimensions using a reference dictionary:

```python
OBJECT_DIMENSIONS = {
    "microwave":    [0.50, 0.36, 0.30],   # meters
    "chair":        [0.50, 0.85, 0.50],
    "mug":          [0.08, 0.10, 0.08],
    "laptop":       [0.33, 0.02, 0.23],
    "refrigerator": [0.70, 1.75, 0.70],
    "sofa":         [2.00, 0.85, 0.90],
    # ... 20+ object types
}
```

Uniform scaling preserves the model's proportions — only the absolute size is corrected.

### Pose Alignment
PCA on the vertex cloud aligns the object's principal axes to world XYZ. The shortest bounding-box extent maps to Z (up). Bottom of bounding box snaps to Z=0 (ground plane). Works correctly for ~80% of AIGC objects; the VLM QA pass catches the rest.

---

## API

```python
from pathlib import Path
from simready.aigc.healer import heal_aigc_mesh
from simready import pipeline

# Stage 1: Heal the raw AIGC mesh
clean_mesh = heal_aigc_mesh(
    input_path=Path("data/aigc_raw/microwave.glb"),
    object_type="microwave",
)

# Stage 2: Run through the physics backend
result = pipeline.run(
    mesh=clean_mesh,
    output_path=Path("output/microwave.usda"),
    enable_vlm=True,
    asset_metadata={
        "aigc:model": "trellis-v1",
        "aigc:prompt": "microwave oven",
    },
)
```

---

## Architecture

```
simready/
  cli.py                   CLI: heal, heal-batch, generate, qc
  pipeline.py              Top-level orchestrator
  aigc/                    AIGC Mesh Healer (frontend)
    mesh_cleaner.py          Geometry healing: components, degenerates, holes
    spatial_aligner.py       PCA pose alignment + scale normalization
    dimensions.py            OBJECT_DIMENSIONS reference dictionary
    healer.py                Orchestrator: load → clean → align → scale
  generation/              AIGC model batch generation
    trellis_client.py        TRELLIS API client
    tripo_client.py          Tripo API client
    prompt_manager.py        Prompt list + manifest tracking
  geometry/                Mesh processing + analytical physics (backend)
    mesh_processing.py       CoACD decomposition, LOD, inertia computation
  materials/               Material mapping (backend)
    material_map.py          VLM inference, 25+ material classes with physics
  usd/                     USD assembly (backend, usd-core only)
    assembly.py              bodies + materials + physics → USD prims
  validation/              Quality assurance
    simready_checks.py       Geometry + material validation, quality score
  acquisition/
    vlm_material.py          Claude API material classification
  semantics/
    classifier.py            Object type → semantic label
  catalog/
    db.py                    SQLite asset catalog
  config/
    settings.py              PipelineSettings, HealerSettings
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `trimesh` | Mesh I/O, geometry healing, PCA, connected components, analytical mass/inertia |
| `numpy` | Geometry math, covariance, eigenvectors, inertia tensors |
| `coacd` | Convex decomposition for collision meshes |
| `usd-core` / `pxr` | OpenUSD authoring — sole USD library |
| `anthropic` | Claude API — VLM material inference + quality assurance |
| `pyrender` or `bpy` | Headless rendering for VLM input and previews |
| `requests` | HTTP client for AIGC model APIs |
| `python-dotenv` | `.env` → `ANTHROPIC_API_KEY` |

**Not used:** `omni.*`, Omniverse Kit, Isaac Sim, PhysX, `cadquery`/`OCP`. Pure Python + `pip install`.

---

## Roadmap

### Done
- CoACD convex decomposition for collision meshes
- Analytical mass/inertia/CoM via trimesh
- VLM material inference (Claude API, 25+ material classes)
- USD assembly with full physics schemas (RigidBodyAPI, MassAPI, CollisionAPI)
- Quality scoring + validation pipeline
- Provenance embedding in USD customData
- SQLite asset catalog

### Phase 1 — AIGC Data Generation & Ingestion
- TRELLIS / Tripo batch generation clients
- Prompt list management + manifest tracking
- 100+ raw AIGC meshes across household/office/workshop categories

### Phase 2 — AIGC Mesh Healer
- `mesh_cleaner.py` — connected components, degenerate removal, watertight repair
- `spatial_aligner.py` — PCA pose alignment + scale normalization to meters
- `healer.py` — end-to-end orchestrator with per-mesh diagnostics

### Phase 3 — Backend Integration
- Pipeline orchestrator connecting healer output to existing physics core
- CLI commands: `heal`, `heal-batch`, `generate-and-heal`
- Batch processing with error isolation and parallel workers

### Phase 4 — VLM Quality Assurance
- Automated visual QA via Claude VLM on rendered previews
- Semantic tagging from VLM analysis
- Feedback loop: VLM flags → re-heal with adjusted parameters

### Phase 5 — Publication
- HuggingFace dataset release
- Benchmarks: drop-test + grasp success + scale correctness (MuJoCo + Isaac Lab)
- Paper: *"From Text Prompt to Physics-Ready"*

---

## License & Commercial Usage

This project is dual-licensed:

* **Academic & Open Source:** Free under the **[AGPLv3 License](LICENSE)**. Derivative works and cloud services must also be open-sourced under AGPLv3.
* **Commercial Use:** Closed-source products, proprietary data generation, or commercial APIs require a **Commercial License**. Contact `@DDBCAAAA`.
