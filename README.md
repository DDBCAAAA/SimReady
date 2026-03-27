# SimReady

**Text prompt → physics-annotated OpenUSD asset, fully automated.**

SimReady generates simulation-ready 3D assets from a plain-English description. An LLM writes CadQuery code, a sandbox compiles it into a STEP file, a VLM critic evaluates the geometry, and the loop self-corrects until the shape passes review. The validated STEP then flows into the physics backend — CoACD collision hulls, analytical mass/inertia, OmniPBR materials — producing a `.usda` file loadable in Isaac Lab, MuJoCo, and ROS out of the box.

Pure Python. `pip install`. No GPU runtime. No external 3D-model API required.

---

## Quick Start

```bash
git clone https://github.com/DDBCAAAA/SimReady.git
cd SimReady
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...

# Generate a bench vise — cast iron, physically correct mass/inertia
simready generate "A bench vise" -o output/vise.usda --material cast_iron

# Kitchen shears — stainless steel
simready generate "A kitchen shears" -o output/shears.usda --material stainless_steel

# Tune the critic threshold and iteration budget
simready generate "A bench vise with a sliding jaw" \
  -o output/vise.usda \
  --material cast_iron \
  --min-confidence 0.80 \
  --max-iterations 7
```

---

## How It Works

Three LLM roles + two deterministic stages, wired into a self-correcting loop:

```
Text prompt
     │
     ▼  Planner + Coder  (Claude)
     │  Blueprint JSON — components, dimensions in meters
     │  CadQuery script — geometry in millimeters (CadQuery/OCC native unit)
     │
     ▼  Sandbox Executor  (subprocess.run)
     │  Runs CadQuery in isolation → .step + .stl
     │  On error: LLM fixes traceback  [inner loop, ≤ 3 retries]
     │
     ▼  Renderer  (trimesh + matplotlib Agg, headless)
     │  2 PNG views: isometric + front
     │
     ▼  VLM Critic  (Claude vision)
     │  PASS (confidence ≥ 0.75)  →  proceed
     │  FAIL or low-confidence    →  LLM revises code  [outer loop, ≤ 5 iters]
     │  Timeout with any PASS     →  use best PASS as fallback
     │
     ▼  Physics Backend  (existing, unchanged)
        CoACD convex decomposition → collision hull set
        trimesh analytical physics → mass, inertia tensor, CoM
        VLM material inference    → PBR properties + density
        usd-core USD assembly     → RigidBodyAPI, MassAPI, CollisionAPI
     │
     ▼
SimReady .usda
  Z-up · meters · physics-complete · engine-agnostic
```

**Key design decisions:**
- Planner and Coder are a single LLM call — blueprint stays coupled to code, saves a round-trip.
- Blueprint is frozen after the first call; only code is revised on subsequent iterations.
- Each revision carries ~6 K tokens (system + blueprint + latest code + critic feedback) — no context bloat.
- CadQuery uses millimeters internally; the STEP reader detects the unit header and applies the correct 0.001 scale to metres. The LLM is told this explicitly.
- Rendering uses trimesh + matplotlib Agg — both already installed, fully headless, no GPU.

---

## Demonstrated Results

### Bench vise — cast iron

```
simready generate "A bench vise" -o output/vise.usda --material cast_iron
```

```
iterations=2  inner_retries=0  quality=0.89  physics=True  mat=cast_iron
```

13 solid bodies tessellated from the generated STEP file:

| Body | Part | Mass |
|---|---|---|
| body_0 | Fixed jaw / base | 2.83 kg |
| body_1 | Sliding jaw body | 3.45 kg |
| body_2, body_6 | Jaw faces | 0.23 kg each |
| body_3, body_4 | Guide rails | 0.22 kg each |
| body_5 | Lead screw | 1.87 kg |
| body_7 | Screw nut | 0.20 kg |
| body_8, body_9 | Handle shaft + collar | 0.15 kg |
| body_10, body_11 | Handle bars | 0.054 kg each |
| body_12 | Mounting plate | 0.086 kg |

**Total: ~9.4 kg cast iron** — physically correct for a 240 × 160 × 100 mm bench vise.

---

## Output Structure

```
/Root                              Z-up · meters · UsdGeom stage
  /Root/Materials/
    cast_iron                      OmniPBR MDL shader
                                   + UsdPhysics.MaterialAPI (friction, restitution)
  /Root/body_0                     UsdGeom.Xform
    RigidBodyAPI
    MassAPI                        mass=2.83 kg  CoM=(x,y,z)  inertia tensor
    MaterialBindingAPI
    /lod0                          Full-res visual mesh
    /lod1                          50% reduced
    /lod2                          25% reduced
    /Collision_0 … N               CoACD convex hull(s) — invisible
      CollisionAPI
    customData:
      simready:qualityScore        0.89
      simready:watertight          true
      simready:physicsComplete     true
      simready:materialConfidence  0.25      ← forced override, not VLM-inferred
      aigc:model                   "cadquery-procedural"
      aigc:prompt                  "A bench vise"
      aigc:iterations              "2"
```

---

## Quality Score

Each asset receives a composite quality score in USD `customData`:

| Component | Weight | Full credit when… |
|---|---|---|
| Watertight mesh | 30% | Closed manifold |
| Physics complete | 40% | Density + friction + restitution + inertia all present |
| Material confidence | 15% | VLM confidence ≥ 0.7 |
| Face density | 15% | ≥ 1,000 faces |

---

## Material Classification

Resolution order for generated assets:

1. **`--material` flag** (recommended) — forces a specific material class and its density for physics. Skips VLM inference entirely.
2. **VLM inference** — Claude API infers material class from object type + mesh geometry context. Returns PBR properties (albedo, roughness, metallic) and physics density. Confidence: 70–95% for well-defined industrial parts.
3. **Fallback** — neutral OmniPBR grey plastic (density = 1050 kg/m³, confidence = 0%).

25+ material classes with full physics properties:
`steel`, `stainless_steel`, `cast_iron`, `aluminum`, `brass`, `bronze`, `copper`, `titanium`, `plastic_abs`, `nylon`, `rubber`, `glass`, `ceramic`, `chrome`, `wood`, and more.

---

## CLI Reference

```bash
# Generate an asset from a text description
simready generate PROMPT [options]

Options:
  -o, --output PATH           Output .usda path
  -m, --material CLASS        Force material class (e.g. cast_iron, nylon)
  --model MODEL               Anthropic model (default: claude-opus-4-6)
  --max-iterations N          Critic-revision cycles (default: 5)
  --max-retries N             Traceback-fix retries per execution (default: 3)
  --min-confidence F          Minimum critic confidence for PASS (default: 0.75)
  --views [isometric front …] Render views sent to the critic
  -c, --config PATH           Pipeline config YAML

# Convert an existing STEP/STL/OBJ file directly (no generation)
simready convert -i part.step -o part.usda

# Batch convert a directory
simready batch --category gear --max-assets 20 --output-dir output/gears/

# Query the asset catalog
simready catalog --query "quality_score > 0.8 AND physics_complete = 1"
```

---

## Architecture

```
simready/
  cli.py                     CLI entry point
  pipeline.py                Physics backend orchestrator
  generation/                Procedural generation agent loop  ← new
    schemas.py                 Blueprint, CriticFeedback, GenerationResult (Pydantic)
    cadquery_reference.py      CadQuery cheat sheet injected into LLM system prompts
    planner.py                 Planner+Coder: prompt → Blueprint + CadQuery code
    executor.py                Sandbox: subprocess.run + inner traceback-fix loop
    renderer.py                Headless render: trimesh + matplotlib Agg → PNG
    critic.py                  VLM Critic: images + Blueprint → PASS/FAIL + corrections
    orchestrator.py            State machine: outer loop + pipeline.run() handoff
  geometry/
    mesh_processing.py         CoACD decomposition, LOD, inertia computation
  materials/
    material_map.py            VLM material inference, 25+ classes with physics
  usd/
    assembly.py                bodies + materials + physics → USD prims (usd-core)
  validation/
    simready_checks.py         Geometry + material validation, quality score
  acquisition/
    vlm_material.py            Claude API material classification
  ingestion/
    step_reader.py             STEP/STP reader (OCP)
    stl_reader.py              STL/OBJ/FBX reader (trimesh)
  config/
    settings.py                PipelineSettings, GenerationSettings
    defaults.yaml              Default configuration
  catalog/
    db.py                      SQLite asset catalog
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `cadquery` | CadQuery 2.x — parametric solid modelling, STEP export |
| `trimesh` | Mesh I/O, analytical mass/inertia/volume, headless rendering |
| `numpy` | Geometry math, inertia tensors |
| `coacd` | Convex decomposition for collision meshes |
| `usd-core` / `pxr` | OpenUSD authoring — sole USD library |
| `anthropic` | Claude API — Planner+Coder, VLM Critic, material inference |
| `matplotlib` | Headless PNG rendering (Agg backend, no display required) |
| `pydantic` | Structured LLM output validation (Blueprint, CriticFeedback) |
| `python-dotenv` | `.env` → `ANTHROPIC_API_KEY` |

**Not used:** `omni.*`, Omniverse Kit, Isaac Sim, PhysX, TRELLIS, Tripo, or any external 3D-model API. The pipeline synthesises geometry from scratch via CadQuery.

---

## License & Commercial Usage

This project is dual-licensed:

- **Academic & Open Source:** Free under the **[AGPLv3 License](LICENSE)**. Derivative works and cloud services must also be open-sourced under AGPLv3.
- **Commercial Use:** Closed-source products, proprietary data generation, or commercial APIs require a **Commercial License**. Contact `@DDBCAAAA`.
