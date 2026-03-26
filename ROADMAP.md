# SimReady: AIGC-to-SimReady Refinery Roadmap

## Strategic Foundation

**Target:** The definitive pipeline for converting raw AIGC 3D outputs into physics-annotated, simulation-ready OpenUSD assets — enabling anyone with a text prompt to produce assets that work in MuJoCo, Isaac Lab, and ROS out of the box.

**Why this, why now:** Text-to-3D models (TRELLIS, Tripo, Meshy, Rodin, InstantMesh) can now generate plausible 3D geometry in seconds. But every output is physically useless — wrong scale, random pose, broken geometry, no mass, no collision, no materials. The gap between "generated mesh" and "simulation-ready asset" is 100% manual labor today. SimReady closes that gap automatically.

**Core insight:** The hard physics problems (convex decomposition, analytical inertia, USD schema assembly) are already solved in our backend. The unsolved problem is the **frontend**: healing the specific pathologies of AIGC meshes before they reach the physics core. That frontend is the entire focus of this roadmap.

**Architectural principle:** Two-stage pipeline. The AIGC Mesh Healer (frontend) produces clean, real-scale, canonically-posed meshes. The Physics Core (backend, existing) takes those clean meshes and produces SimReady USD. The backend is frozen — do not rewrite it.

---

## What Already Exists (Backend — DO NOT REWRITE)

| Component | Module | Status |
|---|---|---|
| CoACD convex decomposition | `geometry/mesh_processing.py` | Done |
| Analytical mass/inertia (trimesh) | `geometry/mesh_processing.py` | Done |
| USD schema assembly (usd-core) | `usd/assembly.py` | Done |
| VLM material inference (Claude API) | `acquisition/vlm_material.py` + `materials/material_map.py` | Done |
| Quality scoring + validation | `validation/simready_checks.py` | Done |
| Provenance embedding in USD | `usd/assembly.py` | Done |
| SQLite asset catalog | `catalog/db.py` | Done |
| 25+ material classes with physics | `materials/material_map.py` | Done |

These modules accept clean `trimesh.Trimesh` objects and produce complete SimReady USD. They are the stable foundation everything below builds on.

---

## Phase 1 — AIGC Data Generation & Ingestion
**Target: April 2026**

### 1a. Text-to-3D batch generation

Write a generation client that takes a list of text prompts and produces raw `.glb`/`.obj` meshes. Target model: **TRELLIS** (open-source, runs locally or via API). Fallback: **Tripo** API.

```bash
# Generate from a prompt list
simready generate --prompts prompts.txt --output-dir ./data/aigc_raw/ --model trellis

# prompts.txt format (one per line):
# microwave oven, household appliance
# office chair with armrests
# ceramic coffee mug
# stainless steel toaster
```

Deliverables:
- `simready/generation/trellis_client.py` — TRELLIS API/local inference wrapper. Input: text prompt. Output: `.glb` file path.
- `simready/generation/tripo_client.py` — Tripo REST API client (API key auth). Fallback generator.
- `simready/generation/prompt_manager.py` — Reads prompt lists (`.txt`, `.csv`), tracks generation status (pending/done/failed), writes manifest file mapping `prompt → output_path → object_type`.
- `prompts.txt` — Seed list of 100+ object prompts spanning household, kitchen, office, workshop categories.

### 1b. Raw mesh ingestion

Support loading all common AIGC output formats:
- `.glb` / `.gltf` — TRELLIS, Tripo, Meshy default output
- `.obj` — universal fallback
- `.fbx` — some commercial generators
- `.ply` — point-cloud-to-mesh pipelines

All loading via `trimesh.load()`. For `.glb` scene graphs with multiple meshes, concatenate into a single `trimesh.Trimesh` (AIGC models are single objects, not assemblies).

**Exit criterion:** 100+ raw AIGC meshes generated and saved to `data/aigc_raw/`. Manifest file tracks prompt, model, object type, and file path for each.

---

## Phase 2 — The AIGC Mesh Healer
**Target: April–May 2026**

This is the core new work. Three modules that clean raw AIGC geometry into backend-ready meshes.

### 2a. `simready/aigc/mesh_cleaner.py` — Geometry Healing

The single most impactful module. AIGC meshes are broken in predictable ways. Fix them all, in order:

```python
def heal(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean an AIGC mesh for physics simulation."""
    # 1. Keep largest connected component
    #    AIGC models produce floating fragments, stray planes, internal geometry.
    #    Split by connectivity, keep the component with the most faces.
    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda c: len(c.faces))

    # 2. Remove degenerate faces (near-zero area)
    #    These cause NaN in normal computation and physics solver divergence.
    areas = mesh.area_faces
    mesh.update_faces(areas > 1e-10)

    # 3. Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    # 4. Merge duplicate vertices
    #    AIGC exporters duplicate vertices at UV seams. Merge within tolerance.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    # 5. Fix winding / normals
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    # 6. Fill holes → attempt watertight
    trimesh.repair.fill_holes(mesh)
    #    Log warning if still non-watertight (backend handles bbox fallback)

    return mesh
```

Metrics to log per mesh:
- Vertices before/after
- Faces before/after
- Components found (how many dropped)
- Watertight after repair? (bool)
- Degenerate faces removed count

### 2b. `simready/aigc/spatial_aligner.py` — Pose Alignment & Scale Normalization

Two problems, one module:

**Pose alignment (PCA):**
```python
def align_pose(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Align mesh principal axes to world XYZ, Z-up."""
    # Covariance matrix of vertex positions
    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue descending: largest variance → X, smallest → Z
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order].T

    # Ensure right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1

    mesh.apply_transform(np.eye(4))  # ... build 4x4 from rotation
    # Snap bottom of bounding box to Z=0
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
    return mesh
```

**Scale normalization:**
```python
OBJECT_DIMENSIONS = {
    # Object type → [W, H, D] in meters (real-world bounding box)
    "microwave":        [0.50, 0.36, 0.30],
    "chair":            [0.50, 0.85, 0.50],
    "office_chair":     [0.65, 1.10, 0.65],
    "mug":              [0.08, 0.10, 0.08],
    "laptop":           [0.33, 0.02, 0.23],
    "toaster":          [0.30, 0.20, 0.18],
    "bottle":           [0.07, 0.25, 0.07],
    "keyboard":         [0.45, 0.03, 0.15],
    "monitor":          [0.55, 0.35, 0.05],
    "desk_lamp":        [0.20, 0.45, 0.20],
    "trash_can":        [0.30, 0.40, 0.30],
    "bookshelf":        [0.80, 1.80, 0.30],
    "nightstand":       [0.50, 0.55, 0.40],
    "washing_machine":  [0.60, 0.85, 0.60],
    "refrigerator":     [0.70, 1.75, 0.70],
    "toilet":           [0.40, 0.40, 0.70],
    "bathtub":          [1.50, 0.50, 0.70],
    "sofa":             [2.00, 0.85, 0.90],
    "dining_table":     [1.20, 0.75, 0.80],
    "bed":              [2.00, 0.60, 1.50],
    # ... extend as categories grow
}

def normalize_scale(mesh: trimesh.Trimesh, object_type: str) -> trimesh.Trimesh:
    """Scale mesh to real-world dimensions in meters."""
    if object_type in OBJECT_DIMENSIONS:
        target = max(OBJECT_DIMENSIONS[object_type])
        current = max(mesh.bounding_box.extents)
        scale_factor = target / current
        mesh.apply_scale(scale_factor)
    else:
        # Fallback: assume current max extent should be 0.3m (generic small object)
        # Or defer to VLM dimension estimation
        pass
    return mesh
```

### 2c. `simready/aigc/healer.py` — Healer Orchestrator

Ties the cleaning steps together into a single callable:

```python
def heal_aigc_mesh(
    input_path: Path,
    object_type: str,
    output_path: Path | None = None,
) -> trimesh.Trimesh:
    """Full AIGC mesh healing pipeline: load → clean → align → scale."""
    mesh = trimesh.load(input_path, force='mesh')
    mesh = clean(mesh)           # mesh_cleaner.heal()
    mesh = align_pose(mesh)      # spatial_aligner.align_pose()
    mesh = normalize_scale(mesh, object_type)  # spatial_aligner.normalize_scale()
    if output_path:
        mesh.export(output_path)
    return mesh
```

**Exit criterion:** Healer processes 100 raw AIGC meshes. Success metrics:
- ≥90% become watertight after healing
- ≥95% have bounding box within 2× of target dimensions
- Zero meshes have degenerate faces or unreferenced vertices
- All meshes are Z-up with bottom on ground plane

---

## Phase 3 — Integration with Existing Physics Core
**Target: May 2026**

### 3a. Pipeline orchestrator

Write the end-to-end flow that connects the healer output to the existing backend:

```python
def aigc_to_simready(
    input_path: Path,
    output_path: Path,
    object_type: str,
    enable_vlm: bool = True,
) -> Path:
    """AIGC mesh → SimReady USD, end-to-end."""
    # Stage 1: Frontend — heal the mesh
    clean_mesh = heal_aigc_mesh(input_path, object_type)

    # Stage 2: Backend — existing physics pipeline
    #   - CoACD convex decomposition
    #   - Analytical mass/inertia via trimesh
    #   - VLM material inference (if enabled)
    #   - USD assembly via usd-core
    return pipeline.run(
        mesh=clean_mesh,
        output_path=output_path,
        enable_vlm=enable_vlm,
        asset_metadata={
            "aigc:model": "trellis",
            "aigc:prompt": object_type,
            "aigc:healer_version": "1.0",
        },
    )
```

### 3b. CLI integration

Extend `cli.py` with AIGC-specific commands:

```bash
# Single mesh
simready heal --input chair.glb --output chair.usda --object-type chair --vlm

# Batch (reads manifest CSV: prompt, file_path, object_type)
simready heal-batch --input-dir ./data/aigc_raw/ --output-dir ./output/ --manifest prompts.csv --vlm

# Generate + heal in one shot
simready generate-and-heal --prompts prompts.txt --output-dir ./output/ --model trellis --vlm
```

### 3c. Batch processing with error isolation

- Per-mesh error isolation — one broken mesh doesn't kill the batch
- Progress: `[47/100] chair_003.glb → chair_003.usda (watertight=true, quality=0.87)`
- Summary report: processed / healed / failed / quality distribution
- `--workers N` for parallel healing (CPU-bound, scales with cores)

**Exit criterion:** 100 AIGC meshes converted end-to-end to SimReady USD. All pass `physicsComplete=true`. ≥80% pass `qualityScore >= 0.7`. Batch pipeline handles failures gracefully.

---

## Phase 4 — VLM Quality Assurance
**Target: May–June 2026**

### 4a. Automated visual QA

After USD generation, render 8-azimuth preview images and pass to Claude VLM for automated verification:

```python
def vlm_quality_check(usd_path: Path, object_type: str) -> QAResult:
    """VLM-based quality assurance on generated SimReady asset."""
    renders = render_previews(usd_path, n_views=8)
    response = claude_api.analyze(
        images=renders,
        prompt=f"""
        This is a 3D asset meant to represent: {object_type}
        Evaluate:
        1. Does the geometry look correct for this object type? (1-10)
        2. Is the pose reasonable (upright, resting on ground)? (yes/no)
        3. Does the scale look physically plausible? (yes/no)
        4. Are there visible artifacts (floating parts, holes, spikes)? (yes/no)
        5. Suggested semantic tags for this object.
        """
    )
    return parse_qa_response(response)
```

### 4b. QA-gated pipeline

Integrate VLM QA as an optional final gate:
- Assets scoring ≥7/10 geometry → pass
- Assets scoring <7/10 → quarantine with VLM reasoning for manual review
- Semantic tags from VLM → written to USD `SemanticsAPI` prims

### 4c. Feedback loop

VLM QA results feed back into the healer:
- If VLM flags "wrong orientation" → re-run pose alignment with alternative heuristic
- If VLM flags "wrong scale" → request VLM dimension estimate, re-scale
- If VLM flags "floating artifacts" → re-run connected component filter with stricter threshold

**Exit criterion:** VLM QA runs on all 100 assets. ≥85% pass on first attempt. Quarantined assets have actionable VLM reasoning. Feedback loop demonstrates measurable improvement on re-processing.

---

## Phase 5 — Dataset Publication & Benchmarks
**Target: June–July 2026**

### 5a. HuggingFace dataset

```python
from datasets import load_dataset
ds = load_dataset("simready/aigc-objects", split="train")
asset = ds[0]  # renders, metadata, USD/URDF/MJCF blobs
```

Dataset card: object type distribution, quality histograms, material class breakdown, AIGC model provenance, healer success rates.

### 5b. Simulation benchmarks

Validate that healed AIGC assets actually work in physics engines. Three experiments, run in both MuJoCo and Isaac Lab:

**Experiment 1 — Drop-test stability:**
Drop all assets onto a flat surface. Measure: penetration depth, settling time, rest-state stability. Compare SimReady assets (healed + physics) vs. raw AIGC meshes (no healing, default physics). Expected: raw meshes explode or fall through; healed assets settle correctly.

**Experiment 2 — Grasp success:**
Franka parallel-jaw grasping on 50 healed assets. 1000 trials per condition. Metric: grasp success rate + post-grasp slip. Compare full-physics assets vs. uniform-default-physics assets. Expected: VLM-inferred friction/density → higher grasp success.

**Experiment 3 — Scale correctness:**
Place healed assets in a reference scene (kitchen counter, desk, shelf). VLM evaluates whether objects appear correctly sized relative to the environment. This validates the dimension dictionary + scale normalization.

```bash
python benchmarks/run_drop_test.py --engine mujoco --assets ./output/
python benchmarks/run_drop_test.py --engine isaac --assets ./output/
python benchmarks/run_grasp_test.py --assets ./output/
python benchmarks/run_scale_eval.py --assets ./output/
```

### 5c. Paper

**Title:** *From Text Prompt to Physics-Ready: Automated Refinement of AIGC 3D Meshes for Robotic Simulation*

**Thesis:** Raw AIGC 3D meshes can be automatically refined into physics-annotated SimReady assets through a systematic pipeline of geometry healing, scale normalization, pose alignment, and analytical physics injection — closing the gap between generative AI and robotic simulation without manual intervention.

**Structure:**
1. **Introduction** — AIGC 3D explosion, sim-to-real gap, manual asset prep bottleneck
2. **Related work** — AIGC 3D models (TRELLIS, Tripo, Meshy), sim-ready standards (NVIDIA SimReady, USD), existing datasets (Objaverse, ShapeNet, Google Scanned Objects)
3. **Pipeline** — Two-stage: AIGC Healer (PCA alignment, scale normalization, geometry healing) + Physics Core (CoACD, trimesh analytics, USD assembly). Pure Python, `pip install`.
4. **The AIGC Mesh Healer** — detailed ablation: what breaks without each step (no cleaning → solver explosion, no scaling → microscopic/giant objects, no alignment → sideways furniture)
5. **Benchmarks** — drop-test, grasp, scale correctness across MuJoCo + Isaac Lab
6. **VLM QA** — automated quality assurance, semantic tagging, feedback loop
7. **Limitations** — PCA alignment fails on symmetric objects, dimension dictionary requires curation, VLM material inference is probabilistic not deterministic, no articulation support

**Target venues:** CoRL 2026, RA-L rolling, ICRA 2027. arXiv preprint immediately.

**Exit criterion:** arXiv live, HuggingFace published, benchmarks reproducible in MuJoCo + Isaac Lab.

---

## Execution Timeline

```
Apr 2026   ████ Phase 1: AIGC generation + ingestion (TRELLIS client, prompt lists, 100 raw meshes)
Apr–May    ████████ Phase 2: AIGC Mesh Healer (mesh_cleaner, spatial_aligner, healer orchestrator)
May 2026   ████████████ Phase 3: Backend integration (pipeline orchestrator, CLI, batch processing)
May–Jun    ████████████████ Phase 4: VLM Quality Assurance (auto QA, feedback loop)
Jun–Jul    ████████████████████ Phase 5: Publication (HuggingFace, benchmarks, paper)
```

---

## Why This Matters

1. **Closes the AIGC-to-simulation gap** — no other tool automatically converts text-to-3D outputs into physics-ready assets
2. **Pure Python, zero dependencies** — `pip install simready`, run on a laptop. No Docker, no GPU, no Omniverse.
3. **Analytically sound physics** — not heuristic mass/inertia; computed from mesh geometry + material density via trimesh
4. **VLM-in-the-loop** — material inference AND quality assurance powered by vision-language models
5. **Engine-agnostic output** — SimReady USD works in Isaac Lab, MuJoCo, ManiSkill, ROS without format conversion
6. **Benchmark proof** — quantitative evidence that healing + physics annotation improves simulation fidelity vs. raw AIGC meshes
7. **Scalable** — text prompt → SimReady USD in minutes, not hours of manual work. Enables synthetic data generation at dataset scale.
