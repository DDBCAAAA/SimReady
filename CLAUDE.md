# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Thinking Framework: First Principles
When approaching any problem, architectural decision, or code generation task, you must strictly apply First Principles Thinking. Do not rely on analogies, standard conventions, or boilerplate solutions unless they are proven to be the most optimal path from the ground up.

Execute your reasoning through the following steps:
1. **Deconstruct to Fundamentals:** Break the problem down into its most basic, undeniable truths and core requirements. What is the absolute bare minimum needed to achieve the objective?
2. **Challenge Assumptions:** Identify and ruthlessly question any hidden assumptions, established industry norms, or "best practices" associated with the prompt. Why are things usually done this way? Is it actually necessary here?
3. **Reconstruct from Scratch:** Synthesize a solution from the ground up using only the foundational truths identified in step 1. Optimize for maximum efficiency, simplicity, and performance.
4. **Expose the Logic:** Briefly outline your deconstruction and reconstruction process before presenting the final code or solution.

Your goal is always to find the most fundamentally sound and direct path to the optimal outcome.

## Strict Directory & Environment Boundaries
1. **Absolute Confinement:** You are strictly forbidden from reading, writing, executing, or modifying ANY files or directories outside of the current working directory (`pwd`).
2. **No Global Paths:** Never use absolute paths that resolve to `/Users/`, `~`, or `/var/`. Always use relative paths (`./` or `src/`).
3. **Environment Isolation:** - For Python: Assume you are running inside a local `.venv`. Always use `python` and `pip`, never `python3` or `pip3` with global flags. Do not modify the system Python environment.
   - For Node.js: Never use `npm install -g`. All package executions must be routed through local `./node_modules/.bin/` or `npm run`.
4. **Output Targeting:** Any generated logs, test data, or output files must be explicitly saved into a `./temp/` or `./output/` folder within this project directory. Create the folder if it does not exist.

## Project Overview

SimReady is an **AIGC-to-SimReady Refinery** — a pure-Python pipeline that takes raw, noisy 3D meshes from generative AI models (Text-to-3D / Image-to-3D like TRELLIS, Tripo, Meshy, Rodin) and refines them into physically accurate, simulation-ready OpenUSD assets. The pipeline runs via `pip install` on a standard machine — no GPU runtime, no simulation engine required.

### The Problem

AIGC 3D models are visually plausible but physically useless. They ship with:
- **Arbitrary scale** — a "microwave" might be 1 unit tall or 0.001 units tall
- **No canonical pose** — random orientation, no consistent up-axis
- **Dirty geometry** — floating artifacts, degenerate faces, non-manifold edges, unreferenced vertices
- **No physics** — zero mass, zero inertia, no collision geometry
- **No materials** — vertex colors at best, no PBR properties, no density

A simulation engine (MuJoCo, Isaac Lab) cannot use these meshes without extensive manual cleanup. SimReady automates this entire cleanup-to-physics pipeline.

### Design Principle: Two-Stage Architecture

```
┌──────────────────────────────────────────────────────┐
│  FRONTEND — AIGC Mesh Healer                         │
│  Ingests raw .glb/.obj from generative models        │
│  Cleans geometry, normalizes scale, aligns pose      │
│  Output: clean, watertight, real-scale mesh          │
└──────────────────────┬───────────────────────────────┘
                       │ clean mesh (trimesh.Trimesh)
                       ▼
┌──────────────────────────────────────────────────────┐
│  BACKEND — Physics Core (EXISTING, DO NOT REWRITE)   │
│  coacd  → convex decomposition for collision hulls   │
│  trimesh → analytical mass, inertia, CoM             │
│  usd-core → USD schema assembly + export             │
│  Output: physics-annotated SimReady .usda             │
└──────────────────────────────────────────────────────┘
```

The **frontend** is the new work. The **backend** already exists and is battle-tested. The frontend's sole job is to produce a clean `trimesh.Trimesh` that the backend can trust.

## Development Commands

```bash
# Install dependencies (pure Python, no GPU runtime required)
pip install -e ".[dev]"

# Heal and convert a single AIGC mesh to SimReady USD
simready heal --input raw_mesh.glb --output asset.usda --object-type microwave

# Batch-heal a directory of AIGC meshes
simready heal-batch --input-dir ./data/aigc_raw/ --output-dir ./output/ --manifest prompts.csv

# Generate raw meshes from text prompts via TRELLIS API
simready generate --prompts prompts.txt --output-dir ./data/aigc_raw/ --model trellis

# VLM quality check on generated USD
simready qc --input asset.usda --render

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name -v

# Lint
ruff check .
ruff format .
```

## Architecture

The project is a **two-stage Python pipeline**: an AIGC mesh healing frontend that cleans generative 3D outputs, feeding into an existing physics backend that produces SimReady USD.

### Pipeline Flow

```
Input: Raw AIGC mesh (.glb, .obj, .fbx)
    │
    ▼
┌─── FRONTEND: AIGC Mesh Healer ────────────────────────────┐
│                                                            │
│  [1. Load & Inspect]                                       │
│      trimesh.load() → scene graph or single mesh           │
│      Log: vertex count, face count, connected components   │
│                                                            │
│  [2. Geometry Healing]  ← mesh_cleaner.py                  │
│      Keep largest connected component (drop artifacts)     │
│      Remove degenerate faces (area < ε)                    │
│      Remove unreferenced vertices                          │
│      Fill small holes → attempt watertight repair          │
│      Merge duplicate vertices (tolerance = 1e-8)           │
│                                                            │
│  [3. Pose Alignment]  ← spatial_aligner.py                 │
│      PCA on vertex cloud → align dominant axes to XYZ      │
│      Heuristic: longest extent → X, second → Y, third → Z │
│      Rotate to Z-up (world convention)                     │
│      Translate so bottom of bbox sits on Z=0 (ground)      │
│                                                            │
│  [4. Scale Normalization]  ← spatial_aligner.py            │
│      Look up real-world dimensions from object type:       │
│        OBJECT_DIMENSIONS = {"microwave": [0.50, 0.36, 0.30], ...} │
│      Compute uniform scale factor: target / current bbox   │
│      Apply scale → mesh is now in meters                   │
│                                                            │
└────────────────────────┬───────────────────────────────────┘
                         │ clean trimesh.Trimesh (watertight, Z-up, meters)
                         ▼
┌─── BACKEND: Physics Core (EXISTING) ──────────────────────┐
│                                                            │
│  [5. Collision Decomposition]  ← coacd                     │
│      CoACD convex decomposition → collision hull set       │
│      NEVER use visual mesh as collision geometry           │
│                                                            │
│  [6. Analytical Physics]  ← trimesh + numpy                │
│      mass = density × mesh.volume                          │
│      inertia = mesh.moment_inertia × density               │
│      center_of_mass = mesh.center_mass                     │
│                                                            │
│  [7. Material Assignment]  ← material_map.py + VLM        │
│      VLM inference (Claude API) from object type + mesh    │
│      → PBR properties (albedo, roughness, metallic)        │
│      → density for physics (steel=7850, plastic=1050, ...) │
│                                                            │
│  [8. USD Assembly]  ← usd-core / pxr                      │
│      UsdGeom: Xform, Mesh, LOD variants                   │
│      UsdPhysics: RigidBodyAPI, MassAPI, CollisionAPI       │
│      UsdShade: OmniPBR material binding                    │
│      customData: provenance, quality, AIGC source info     │
│                                                            │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
Output: SimReady .usda — physics-annotated, engine-agnostic
    Loadable in: Isaac Lab, MuJoCo (MJCF export), ROS (URDF export)
```

### Core Tech Stack

| Layer | Library | Role |
|---|---|---|
| **Mesh I/O & healing** | `trimesh` + `numpy` | Load `.glb`/`.obj`/`.fbx`. Connected-component analysis, degenerate-face removal, hole filling, watertight repair. PCA for pose alignment. All geometry math. |
| **Convex decomposition** | `coacd` | Collision-Aware Convex Decomposition. Generates airtight convex hulls from non-convex visual meshes. Every asset gets CoACD hulls — never raw visual geometry as collision. |
| **USD authoring** | `usd-core` (`pxr`) | Sole library for composing and exporting `.usd`/`.usda` files. Writes all `UsdGeom`, `UsdPhysics`, and `UsdShade` schemas directly. No Omniverse Kit, no `omni.*`. |
| **Material inference** | `anthropic` (Claude API) | VLM-based material + semantic classification. Infers PBR properties and physics density from object type, rendered preview, and mesh geometry context. |
| **Headless rendering** | `pyrender` or `bpy` | Lightweight multi-view preview renders for VLM input and quality assurance. No GPU simulation runtime. |
| **AIGC generation** | `requests` / model-specific SDK | Batch text-to-3D generation via TRELLIS, Tripo, or other open-source AIGC APIs. |

### AIGC Mesh Healer Rules (CRITICAL)

These rules define the frontend's contract. A mesh that exits the healer MUST satisfy all of them before entering the physics backend.

#### Rule 1: Geometry Healing
Raw AIGC meshes are noisy. The healer performs, in order:
1. **Largest connected component** — AIGC models often produce floating artifacts (stray vertices, disconnected chunks). Keep only the largest connected component by face count. Drop everything else.
2. **Degenerate face removal** — Remove faces with near-zero area (`area < 1e-10`). These cause NaN in normal computation and physics solvers.
3. **Unreferenced vertex removal** — After component filtering and face removal, purge orphan vertices.
4. **Duplicate vertex merge** — Merge vertices within `1e-8` tolerance. AIGC exporters often duplicate vertices at seams.
5. **Hole filling** — Attempt watertight repair via `trimesh.repair.fill_holes()`. If the mesh remains non-watertight after repair, log a warning and proceed with bounding-box volume approximation in the physics backend.

#### Rule 2: Scale Normalization
AIGC models have no concept of real-world scale. A "chair" might be 0.001 units or 100 units tall. The healer MUST map the mesh to physically correct dimensions in meters.

Strategy:
- Maintain a **reference dimension dictionary** mapping object types to real-world bounding-box extents in meters:
  ```python
  OBJECT_DIMENSIONS = {
      "microwave":    [0.50, 0.36, 0.30],  # W × H × D in meters
      "chair":        [0.50, 0.85, 0.50],
      "mug":          [0.08, 0.10, 0.08],
      "laptop":       [0.33, 0.02, 0.23],
      "toaster":      [0.30, 0.20, 0.18],
      "bottle":       [0.07, 0.25, 0.07],
      # ... extend per object category
  }
  ```
- Compute **uniform scale factor**: `scale = max(target_extents) / max(current_extents)`. Uniform scaling preserves aspect ratio — the AIGC model's proportions are trusted; only the absolute size is wrong.
- If the object type is not in the dictionary, fall back to VLM-based dimension estimation or a configurable default scale.

#### Rule 3: Pose Alignment
AIGC models have random orientation. The healer MUST produce a canonical Z-up resting pose.

Strategy:
1. **PCA alignment** — Run Principal Component Analysis on the vertex cloud via `numpy.linalg.eigh` on the covariance matrix. The three eigenvectors define the object's principal axes. Map: longest extent → X, second → Y, shortest → Z. This is a heuristic — it works well for elongated objects but may need refinement for symmetric ones.
2. **Z-up rotation** — After PCA, ensure the shortest bounding-box dimension is aligned with Z (up). For objects that "rest" on a flat base (chairs, mugs), the base should sit on Z=0.
3. **Ground-plane snap** — Translate the mesh so `min(vertices[:, 2]) = 0`. The object sits on the ground plane, ready for drop-test validation.

These heuristics handle ~80% of AIGC objects correctly. For ambiguous cases (spheres, cubes), the downstream VLM QA pass catches misalignments.

### Physics Rules (Backend — EXISTING, DO NOT MODIFY)

These rules are absolute. The backend enforces them on whatever the frontend delivers:

1. **Collision meshes**: ALWAYS generated via `coacd`. Never use visual meshes directly as collision geometry. Non-convex collision geometry causes solver explosions in any physics engine.
2. **Mass properties**: Computed analytically via `trimesh`. For watertight meshes: `mass = density × mesh.volume`, `inertia = mesh.moment_inertia * density`. For non-watertight: bounding-box volume approximation.
3. **No engine at generation time**: Physics properties are written into USD schemas by `usd-core`. No `SimulationApp`, no PhysX, no `omni.*`. Generated USD is validated downstream in target engines.

### Material Mapping (AIGC Path)

AIGC meshes have zero embedded material data. Resolution order for AIGC assets:

1. **VLM inference** (primary path) — Claude API classifies material from object type + rendered preview + mesh geometry context. Returns: material class, PBR properties (albedo, roughness, metallic), physics density, confidence score. Typical confidence: 70–95%.
2. **Object-type hint** — If VLM is disabled or unavailable, map object type to a default material class: `"microwave" → "steel"`, `"chair" → "plastic_abs"`, `"mug" → "ceramic"`.
3. **Fallback** — Neutral OmniPBR grey plastic (density=1050, confidence=0%).

The pipeline tracks provenance via `simready:materialConfidence` and `simready:vlmReasoning` in USD customData.

### Module Layout

```
simready/
  cli.py                   # CLI: heal, heal-batch, generate, qc
  pipeline.py              # Top-level orchestrator
  quality_gate.py          # Material confidence gate + quarantine
  aigc/                    # AIGC Mesh Healer (NEW — the core frontend)
    mesh_cleaner.py        #   Geometry healing: components, degenerates, holes, watertight
    spatial_aligner.py     #   PCA pose alignment + scale normalization to meters
    dimensions.py          #   OBJECT_DIMENSIONS reference dictionary
    healer.py              #   Orchestrator: load → clean → align → scale → export clean mesh
  generation/              # AIGC model batch generation
    trellis_client.py      #   TRELLIS API client for text-to-3D
    tripo_client.py        #   Tripo API client
    prompt_manager.py      #   Prompt list management + manifest tracking
  geometry/                # Mesh processing + analytical physics (EXISTING)
    mesh_processing.py     #   clean, center_at_com, scale_to_meters, LOD, CoACD, inertia
  materials/               # Material mapping (EXISTING)
    material_map.py        #   VLM material inference, 25+ material classes
  usd/                     # USD assembly (EXISTING, usd-core only)
    assembly.py            #   create_stage(): bodies, materials → USD prims
  validation/              # Quality assurance (EXISTING)
    simready_checks.py     #   Geometry + material validation, quality score
  acquisition/             # Asset acquisition (EXISTING — retained for VLM)
    vlm_material.py        #   Claude API material classification
  semantics/               # Semantic classification (EXISTING)
    classifier.py          #   Object type → semantic label
  catalog/                 # Asset tracking (EXISTING)
    db.py                  #   SQLite catalog
  config/                  # Pipeline configuration
    settings.py            #   PipelineSettings, HealerSettings, etc.
tests/
data/                      # Assets (gitignored)
  aigc_raw/                #   Raw AIGC meshes (.glb, .obj) from generative models
  aigc_healed/             #   Cleaned meshes after healer pass
  catalog.db               #   SQLite catalog
output/                    # Generated SimReady USDs
```

### Provenance & Licensing

Every generated USD asset embeds in `/Root` customData:
- `_IP_Signature`, `_Generation_Time`, `_License` (AGPLv3), `_Source` (GitHub repo URL)
- `aigc:model` — which generative model produced the raw mesh (e.g., "trellis-v1", "tripo-2.0")
- `aigc:prompt` — the text prompt used for generation
- `aigc:healer_version` — version of the mesh healer that processed it
- `aigc:scale_source` — how scale was determined ("dimension_dict", "vlm_estimate", "default")
- `simready:materialConfidence`, `simready:vlmReasoning` — material inference provenance

## Key Dependencies

| Package | Purpose |
|---|---|
| `trimesh` | Mesh I/O, geometry healing, PCA alignment, connected components, analytical mass/inertia/volume. The workhorse of both frontend and backend. |
| `numpy` | Geometry math — covariance matrix, eigenvectors for PCA, inertia tensor computation |
| `coacd` | Convex decomposition for collision meshes (backend) |
| `usd-core` (`pxr`) | OpenUSD authoring — sole USD library, writes all schemas (backend) |
| `anthropic` | Claude API for VLM material inference + optional QA verification |
| `pyrender` or `bpy` | Lightweight headless rendering for VLM input and preview images |
| `requests` | HTTP client for AIGC model APIs (TRELLIS, Tripo) |
| `tqdm` | Progress bars for batch operations |

**Explicitly not used:** `omni.*`, Omniverse Kit SDK, Isaac Sim, `SimulationApp`, PhysX Python bindings, `cadquery`/`OCP` (no CAD parsing needed for AIGC path). The pipeline is pure Python + `pip install`.

## SimReady Asset Standards

Output assets follow NVIDIA SimReady conventions:
- Z-up axis (aligned by the healer's PCA + rotation step)
- Meters as the default unit (`metersPerUnit = 1.0`; enforced by the healer's scale normalization)
- Physics schema applied to ALL assets (`UsdPhysics.RigidBodyAPI`, `MassAPI`, `CollisionAPI`)
- Collision meshes from CoACD — never raw visual geometry
- Semantic labels on prims (`SemanticsAPI`) — VLM-inferred from object type
- OmniPBR materials via `UsdShade` — VLM-inferred PBR properties
- Per-asset quality metadata: `simready:qualityScore`, `simready:watertight`, `simready:physicsComplete`, `simready:materialConfidence`
