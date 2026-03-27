# SimReady Roadmap

## What SimReady Is Now

SimReady generates physics-annotated OpenUSD assets from a plain-English description. The approach is procedural synthesis: an LLM writes CadQuery code, a sandbox compiles it, a VLM critic evaluates the rendered geometry, and the loop self-corrects until the shape passes review. The validated STEP file then flows into the existing physics backend — CoACD, trimesh analytics, usd-core — to produce a complete SimReady `.usda`.

This is a deliberate departure from the original AIGC-healer approach (ingesting raw `.glb`/`.obj` from Text-to-3D models and fixing their pathologies). Instead of healing broken geometry, SimReady generates correct geometry from scratch. The output is watertight by construction, in the right unit system, and assembled from semantically meaningful CadQuery primitives — which gives the physics backend far cleaner input than healed generative meshes.

---

## What Is Done

### Physics Backend — Stable, Unchanged

| Component | Module | Status |
|---|---|---|
| CoACD convex decomposition | `geometry/mesh_processing.py` | ✓ Done |
| Analytical mass / inertia / CoM (trimesh) | `geometry/mesh_processing.py` | ✓ Done |
| USD schema assembly (usd-core) | `usd/assembly.py` | ✓ Done |
| VLM material inference (Claude API) | `acquisition/vlm_material.py` + `materials/material_map.py` | ✓ Done |
| 25+ material classes with physics properties | `materials/material_map.py` | ✓ Done |
| Quality scoring + validation | `validation/simready_checks.py` | ✓ Done |
| Provenance embedding in USD customData | `usd/assembly.py` | ✓ Done |
| LOD generation (trimesh mesh simplification) | `geometry/mesh_processing.py` | ✓ Done |
| SQLite asset catalog | `catalog/db.py` | ✓ Done |
| STEP / STL / OBJ ingestion | `ingestion/step_reader.py`, `stl_reader.py` | ✓ Done |

### Procedural Generation Agent Loop — Implemented

| Component | Module | Status |
|---|---|---|
| Pydantic schemas: Blueprint, CriticFeedback, GenerationResult | `generation/schemas.py` | ✓ Done |
| CadQuery reference cheat sheet (injected into LLM prompts) | `generation/cadquery_reference.py` | ✓ Done |
| Planner+Coder: prompt → Blueprint + CadQuery script | `generation/planner.py` | ✓ Done |
| Sandbox executor: subprocess.run + inner traceback-fix loop | `generation/executor.py` | ✓ Done |
| Headless renderer: trimesh + matplotlib Agg → PNG | `generation/renderer.py` | ✓ Done |
| VLM Critic: rendered images + blueprint → PASS / FAIL | `generation/critic.py` | ✓ Done |
| State machine orchestrator: outer loop + fallback + pipeline handoff | `generation/orchestrator.py` | ✓ Done |
| `simready generate` CLI subcommand | `cli.py` | ✓ Done |
| GenerationSettings in config (model, iterations, confidence threshold) | `config/settings.py` | ✓ Done |

**Demonstrated on:**
- `simready generate "A bench vise" --material cast_iron` → 13-body STEP, 9.4 kg cast iron, quality=0.89, physics=True, 2 iterations
- `simready generate "A kitchen shears"` → 3-body STEP, geometry validated, fallback at best-PASS-of-5 used

---

## Phase 2 — Geometry Depth  (next)

The current generator handles solid bodies well. Phase 2 targets geometry that requires more advanced CadQuery patterns.

### 2a. Articulated assemblies

Multi-body objects with joints (hinges, sliders, pivots). The Blueprint `joints` field already carries joint intent — wire it through to USD `UsdPhysics.RevoluteJoint` / `PrismaticJoint` prims.

Target objects: scissors, pliers, box lids, cabinet doors, drawer slides.

Deliverables:
- Joint-aware prompt templates in `planner.py` that instruct the LLM to produce separate bodies per rigid link and encode joint intent in the blueprint.
- Assembly-to-per-body STEP splitter in `executor.py` (CadQuery `.toCompound()` → split by body).
- Joint prim assembly in `usd/assembly.py` — map `JointSpec` to `UsdPhysics` joint schemas.
- CLI: `simready generate "A pair of scissors" --articulated`

### 2b. Profile-extruded and revolved geometry

Complex cross-sections that benefit from sketch + extrude / revolve patterns rather than boolean operations on primitives. The cheat sheet already covers these; success depends on the LLM using them consistently.

Target objects: flanged pipe, gear teeth, I-beams, threaded fasteners (simplified), wrenches.

Deliverables:
- Expanded CadQuery examples for profile-based patterns in `cadquery_reference.py`.
- Critic prompt additions: check that revolved geometry is axially symmetric, extruded geometry has consistent cross-section.

### 2c. Confidence calibration

The critic currently converges to 0.62–0.72 for thin or topologically complex objects (shears, scissors) regardless of revision count. This is a calibration problem, not a geometry problem — the critic should be told what "PASS" means for objects that are inherently thin or have open loops.

Deliverables:
- Object-class-aware critic system prompt addendum: pass different evaluation criteria for "thin / bladed" vs. "solid / blocky" objects.
- `--object-class` CLI flag to route critic prompts (default: "solid").

---

## Phase 3 — Dataset Generation at Scale

The agent loop produces one asset per run. Phase 3 makes it a dataset factory.

### 3a. Batch generation

```bash
simready generate-batch \
  --prompts prompts.txt \
  --output-dir output/dataset/ \
  --material-map material_map.csv \
  --workers 4
```

`prompts.txt` — one prompt per line. `material_map.csv` — maps prompt keywords to forced material class. `--workers` — parallel subprocesses, each running an independent agent loop.

Deliverables:
- `simready/generation/batch_generator.py` — reads prompt list, dispatches `orchestrator.generate()` calls in a `ProcessPoolExecutor`, writes per-asset JSON logs.
- Resume support: skip prompts that already have a corresponding `.usda` in the output dir.
- Summary report: total / succeeded / fallback / failed, quality histogram, material breakdown.

### 3b. Prompt library

A curated `prompts/` directory covering the categories most useful for robot manipulation research:

```
prompts/
  household_tools.txt       hammer, screwdriver, wrench, pliers, scissors
  kitchen_objects.txt       mug, bowl, bottle, spatula, ladle, colander
  office_objects.txt        stapler, tape dispenser, hole punch, binder clip
  workshop_parts.txt        vise, clamp, bolt, nut, washer, bearing
  containers.txt            box, crate, cylinder, bucket, jar, can
```

Target: 200 prompts, ≥ 160 successful assets (80% pass rate), all with physics-complete USD.

### 3c. HuggingFace dataset release

```python
from datasets import load_dataset
ds = load_dataset("simready/procedural-objects", split="train")
asset = ds[0]
# keys: prompt, material_class, quality_score, watertight, physics_complete,
#       usd_bytes, step_bytes, renders (list of PNG), blueprint (JSON)
```

Dataset card: quality distribution, material class breakdown, per-category success rates, generation iteration counts.

---

## Phase 4 — Benchmarks & Publication

### 4a. Simulation benchmarks

Three experiments, run in both MuJoCo and Isaac Lab, to validate that procedurally generated assets work correctly in physics engines.

**Drop-test stability:** Drop all assets from 0.5 m height onto a flat surface. Measure penetration depth at rest, time to settle, rest-state stability score. Compare: SimReady generated (correct mass, CoACD hulls) vs. same STEP with default physics (uniform 1 kg, convex hull approximation). Expected: SimReady assets settle correctly; default-physics assets tip, spin, or tunnel.

**Grasp success:** Franka parallel-jaw grasping on 50 generated assets. 500 trials per object. Metric: grasp success rate + post-grasp slip at 5 s. Compare: VLM-inferred friction/density (SimReady) vs. uniform defaults. Expected: correct density → correct finger force → higher success rate.

**Scale sanity:** Place generated assets in a reference scene (desk, kitchen counter, workbench). VLM evaluates whether object sizes are plausible relative to the scene. Validates that the CadQuery blueprint dimensions are realistic.

```bash
python benchmarks/drop_test.py --engine mujoco --assets output/dataset/
python benchmarks/drop_test.py --engine isaac --assets output/dataset/
python benchmarks/grasp_test.py --assets output/dataset/
python benchmarks/scale_eval.py --assets output/dataset/
```

### 4b. Paper

**Title:** *SimReady: Procedural Physics-Annotated 3D Asset Generation via LLM-Driven CadQuery Synthesis*

**Thesis:** Precise, physics-correct 3D assets can be generated from plain-English descriptions by treating the problem as agentic code synthesis: an LLM writes CadQuery geometry code, a VLM critic evaluates rendered views in a self-correcting loop, and the validated solid is handed to a physics backend that injects collision hulls, mass properties, and USD schemas. This approach produces watertight, correctly-scaled geometry without requiring external 3D-model APIs, point clouds, or mesh healing.

**Target venues:** CoRL 2026, RA-L rolling, ICRA 2027. arXiv preprint immediately after dataset release.

---

## Execution Timeline

```
Apr 2026   ████████████████████ Phase 1 DONE — agent loop, physics backend, CLI
Apr–May    ████ Phase 2 — articulated assemblies, profile geometry, critic calibration
May–Jun    ████████ Phase 3 — batch generation, prompt library, dataset release
Jun–Jul    ████████████ Phase 4 — benchmarks (MuJoCo + Isaac Lab), paper
```

---

## Why Procedural Synthesis Instead of Healing

The original AIGC-healer design aimed to fix the pathologies of Text-to-3D model outputs (arbitrary scale, random pose, broken geometry). That approach has a structural problem: the healer can only approximate — it can't know what a "microwave" should look like, so it applies a dimension dictionary and PCA alignment and hopes for the best. The geometry is still whatever the generative model produced.

Procedural synthesis inverts the problem. Instead of fixing broken geometry, the LLM designs correct geometry. The CadQuery script is a deterministic CAD program — it produces watertight solids by construction, with dimensions that mean exactly what the prompt said. The VLM critic closes the loop on geometric plausibility. The result is cleaner input to the physics backend, fewer CoACD decomposition failures, and physically meaningful mass values from the first run.

The tradeoff is that complex topology (open loops, thin blades, fine threads) is harder to generate correctly than to import. Phase 2 addresses this by expanding the geometry patterns the agent can reliably produce.
