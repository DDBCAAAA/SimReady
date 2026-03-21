# SimReady

**Automated CAD/CAE → OpenUSD pipeline for physics-based simulation and world model training.**

SimReady converts raw mechanical STEP files into simulation-ready USD assets that meet the material fidelity, physics completeness, and semantic labeling standards required by NVIDIA Omniverse, Isaac Lab, and GR00T.

---

## What It Does

Most CAD-to-USD tools produce geometry-only assets: correct shape, wrong physics, no materials, no semantic meaning. SimReady treats material fidelity as the primary quality metric and produces assets that can drop straight into a physics simulator.

```
STEP / STL / FEA mesh
        │
        ▼  Ingestion (OCC / trimesh)
        │  ─ STEP product names extracted via XDE
        │  ─ Multi-body assemblies preserved
        ▼  Geometry Processing
        │  ─ Mesh cleaning + watertightness repair
        │  ─ Center-of-mass pivot normalization
        │  ─ mm → m unit conversion
        │  ─ 3-level LOD VariantSet (100% / 50% / 25%)
        ▼  Material Mapping
        │  ─ Regex/keyword classifier (zero-cost, deterministic)
        │  ─ VLM fallback (Claude API, single call per part)
        │  ─ Material confidence gate (≥ 0.8 to proceed)
        ▼  USD Assembly
        │  ─ MDL / OmniPBR shader (diffuse, roughness, metallic)
        │  ─ UsdPhysics: RigidBodyAPI, MassAPI, MaterialAPI
        │  ─ Convex collision mesh (CoACD for gears/cams)
        │  ─ Exact mass + inertia tensor from trimesh
        │  ─ SimReady semantic label on every prim
        ▼
OpenUSD (.usda / .usdc)  —  Omniverse-ready
```

---

## Output Structure

Every converted asset follows the NVIDIA SimReady USD schema:

```
/Root                              Z-up · meters · UsdGeom stage
  /Root/Materials/
    steel                          OmniPBR MDL shader
                                   + UsdPhysics.MaterialAPI (friction, restitution)
  /Root/<PartName>                 UsdGeom.Xform
    RigidBodyAPI                   ← simulated rigid body
    MassAPI                        ← exact mass, CoM, inertia tensor
    MaterialBindingAPI             ← bound to /Root/Materials/steel
    lod VariantSet
      lod0                         full detail mesh (e.g. 208 k faces)
      lod1                         50 % decimation
      lod2                         25 % decimation
    /Collision_0                   convex hull mesh (invisible)
      CollisionAPI                 ← used by PhysX
    customData:
      simready:semanticLabel       "structural:flange"
      simready:partName            "DN15_Stamped_Flange"
      simready:qualityScore        0.958
      simready:watertight          true
      simready:physicsComplete     true
      simready:materialConfidence  0.72
```

---

## Quality Score

Each asset receives a composite quality score written as USD `customData`:

| Component | Weight | Full credit when… |
|---|---|---|
| Watertight mesh | 30 % | trimesh reports a closed manifold |
| Physics complete | 40 % | density + static/dynamic friction + restitution all present |
| Material confidence | 15 % | ≥ 1.0 (comes from CAE data or high-confidence VLM) |
| Face density | 15 % | ≥ 1 000 faces |

Assets below **0.8 material confidence** are quarantined before USD generation and logged to `output/low_confidence_assets.log`.

---

## Material Classification

Materials are resolved in priority order:

1. **CAE file** — if the source contains Young's modulus, density, or thermal conductivity the material class is derived directly.
2. **Regex / keyword pass** — part name matched against a deterministic rule set (e.g. `"_M6"` → steel, `"6061"` → aluminum). Zero cost, always runs first.
3. **VLM fallback** — when name evidence is ambiguous, a single Claude API call classifies material *and* semantic label simultaneously. Filename stem is used instead of generic body names (`body_0`, `solid_1`) so the model sees `"ISO4032_Hex_Nut_M6"` rather than a placeholder.

Supported material classes: `steel`, `stainless_steel`, `aluminum`, `brass`, `bronze`, `copper`, `titanium`, `cast_iron`, `plastic_abs`, `plastic_nylon`, `plastic_pom`, `rubber`, `glass`, `ceramic`.

---

## Semantic Taxonomy

Every prim is labeled using the SimReady taxonomy `<category>:<subcategory>`:

| Category | Labels |
|---|---|
| `fastener` | `bolt` · `nut` · `washer` · `rivet` · `pin` |
| `mechanical` | `gear` · `bearing` · `shaft` · `spring` · `pulley` · `cam` |
| `structural` | `plate` · `bracket` · `beam` · `frame` · `flange` · `enclosure` |
| `fluid_system` | `pipe` · `valve` · `fitting` · `nozzle` |
| `electrical` | `connector` · `housing` |
| `industrial_part` | `component` _(fallback)_ |

Labels feed directly into physics decomposition decisions — gears and cams automatically receive full CoACD convex decomposition instead of a single convex hull, preserving tooth/bore geometry for contact simulation.

---

## Dataset

157 open-source STEP files organized across 8 mechanical categories:

```
data/step_files/
  brackets_plates/          L-brackets, gantry plates, aluminum panels
  connectors_fittings/      pipe couplings, hose fittings, tube connectors
  fasteners/                ISO hex nuts, socket screws, flanged bolts
  gears/                    spur gears, bevel gears, worm gears
  housings_enclosures/      motor housings, bearing blocks, covers
  misc_mechanical/          springs, pulleys, cams
  shafts_bearings/          shafts, bearing races, spindles
  valves_pipe/              DN15 ball valves, stamped flanges, gate valves
```

Sources: FreeCAD community models, NIST MBE PMI dataset, ABC Dataset (MIT license).

---

## Quick Start

```bash
# Install
git clone https://github.com/DDBCAAAA/SimReady.git
cd SimReady
pip install -e ".[dev]"

# Convert a single STEP file
simready convert \
  --input  data/step_files/fasteners/ISO4032_Hex_Nut_M6.step \
  --output output/ISO4032_Hex_Nut_M6.usda

# Enable VLM-enhanced material + semantic labeling
# Requires ANTHROPIC_API_KEY in .env
simready convert \
  --input  data/step_files/fasteners/ISO4032_Hex_Nut_M6.step \
  --output output/ISO4032_Hex_Nut_M6.usda \
  --config temp/vlm_config.yaml

# Search and download more STEP files
simready acquire "spur gear" --max-results 20
simready acquire "ball valve" --sources github
simready catalog          # list downloaded assets

# Batch convert all assets (4 parallel workers)
python temp/convert_all.py
```

**VLM config** (`temp/vlm_config.yaml`):

```yaml
materials:
  enable_vlm: true
  vlm_model: "claude-haiku-4-5"   # ~$0.0008 / part
  vlm_max_calls: 200               # hard cap ≈ $0.16 total

validation:
  enable_confidence_gate: false    # set true for production
```

---

## API

```python
from pathlib import Path
from simready import pipeline

result = pipeline.run(
    input_path=Path("part.step"),
    output_path=Path("output/part.usda"),
    config_path=Path("temp/vlm_config.yaml"),
    material_overrides={"*": "aluminum"},   # optional: force material
)

# result dict
{
    "face_count":           208356,
    "quality_score":        0.958,
    "watertight":           True,
    "physics_complete":     True,
    "material_confidence":  0.72,
    "material_class":       "steel",
}
```

---

## Architecture

```
simready/
  pipeline.py          Top-level orchestrator (6 stages)
  cli.py               CLI: convert · acquire · catalog
  acquisition/
    sources.py         Pluggable STEP source registry (@register_source)
    freecad_source.py  FreeCAD community models
    nist_source.py     NIST MBE PMI dataset
    abc_dataset.py     ABC Dataset (~1M CAD models, MIT)
    vlm_material.py    Claude API material + semantic classifier
  ingestion/
    step_reader.py     OCC/XDE STEP parser → CADBody[]
    stl_reader.py      trimesh STL/OBJ/PLY reader
  geometry/
    mesh_processing.py tessellation, cleanup, LOD, CoACD decomposition
  materials/
    material_map.py    CAEMaterial → MDLMaterial (PBR + physics props)
  semantics/
    classifier.py      Keyword → SimReady taxonomy label
  usd/
    assembly.py        OpenUSD stage builder (geometry + materials + physics)
  validation/
    simready_checks.py quality score, geometry and material validation
  quality_gate.py      Confidence gate (quarantine below 0.8)
  config/
    settings.py        PipelineSettings dataclass
    defaults.yaml      Default pipeline config
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `usd-core` / `pxr` | OpenUSD Python bindings |
| `trimesh` | Mesh I/O, watertightness, mass properties |
| `numpy` | Geometry / linear algebra |
| `OCP` (pythonocc-core) | STEP / IGES CAD parsing via OpenCASCADE |
| `fast-simplification` | LOD mesh decimation |
| `anthropic` | Claude API for VLM material classification |
| `python-dotenv` | `.env` → `ANTHROPIC_API_KEY` |
| `aiohttp` | Async HTTP for acquisition agent |

---

## Roadmap

### Done ✅
- Multi-body STEP ingestion with XDE product names
- Mesh cleaning, CoM pivot normalization, mm → m scaling
- 3-level LOD VariantSets
- Regex + VLM two-pass material classifier
- VLM returns `material_class` + `confidence` + `semantic_label` in one call
- Filename stem fallback for generic FreeCAD body names (`body_0` → `ISO4032_Hex_Nut_M6`)
- Exact mass + inertia tensor from watertight mesh + material density
- Convex hull collision mesh; CoACD decomposition for gears and cams
- UsdPhysics: RigidBodyAPI, MassAPI, MaterialAPI (friction + restitution)
- Material confidence gate (quarantine below 0.8)
- Quality score written as USD `customData`
- 193-test suite (all mocked, no network required)
- Batch acquisition from GitHub, NIST, ABC Dataset

### In Progress 🔧
- **P3.1** STEP assembly tree (`NEXT_ASSEMBLY_USAGE_OCCURRENCE`) → nested USD Xforms with transforms
- **P3.2** Full CoACD tuning per-label (tooth count → hull count heuristic for gears)

### Planned 📋
- **P4.1** Articulation joints (`KINEMATIC_JOINT`) → `UsdPhysics.RevoluteJoint` / `PrismaticJoint`
- **P4.2** FEA result overlay — stress / strain fields as USD primvars for training signal
- **P4.3** Omniverse Kit extension for live preview and quality dashboard
- **P4.4** Batch IGES + STEP AP242 support

---

## License

MIT — see [LICENSE](LICENSE).

Assets in `data/` are sourced from open-licensed repositories (FreeCAD community, NIST MBE PMI, ABC Dataset). Individual file licenses are tracked in `data/catalog.json`.
