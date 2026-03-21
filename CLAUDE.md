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

SimReady is an automated agent pipeline that converts CAD/CAE source files into high-accuracy OpenUSD assets suitable for training world models. Assets must meet material fidelity standards required for physics-based simulation in NVIDIA Omniverse.

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Convert a CAD file to Sim-Ready USD
simready convert --input path/to/asset.step --output path/to/output.usd

# Search and download open-source STEP files
simready acquire "gear" --max-results 10
simready acquire "bracket" --search-only        # search without downloading
simready acquire "automotive" --sources github   # specific source only
simready catalog                                 # view downloaded assets

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name -v

# Lint
ruff check .
ruff format .
```

## Architecture

The project is a **single Python pipeline** with the following stages:

### Phase 1: CAD/CAE → OpenUSD Conversion

```
Input (CAD/CAE)
    │
    ▼
[Ingestion] — parse STEP, IGES, STL, FEA mesh formats
    │
    ▼
[Geometry Processing] — tessellation, mesh cleanup, LOD generation
    │
    ▼
[Material Mapping] — extract material properties from CAE, map to PBR/MDL
    │
    ▼
[USD Assembly] — compose prims, apply schemas, write .usd/.usda
    │
    ▼
Output (OpenUSD) — Omniverse-compatible Sim-Ready asset
```

### Key Concepts

- **Sim-Ready**: Assets conform to NVIDIA SimReady standards — physically accurate geometry, semantically labeled prims, PBR/MDL materials, correct physics properties.
- **OpenUSD**: All output uses the OpenUSD format. Use `pxr` (USD Python bindings) for reading/writing USD. Prefer `.usda` (ASCII) for debugging, `.usd`/`.usdc` for production.
- **MDL (Material Definition Language)**: NVIDIA's material language used in Omniverse. Material fidelity is the primary quality metric — CAE material properties (Young's modulus, density, roughness, etc.) must map correctly to MDL shaders.
- **Omniverse**: Target runtime. Use `omni.usd` and Kit SDK APIs where needed for validation and preview.

### Module Layout

```
simready/
  pipeline.py        # Top-level orchestrator
  cli.py             # CLI with subcommands: convert, acquire, catalog
  acquisition/       # Automated STEP file search & download (GitHub, ABC Dataset)
  ingestion/         # CAD/CAE format readers (STEP, IGES, STL, FEA)
  geometry/          # Mesh processing, tessellation, LOD
  materials/         # CAE-to-MDL material mapping logic
  usd/               # USD assembly, schema application, export
  validation/        # Material fidelity checks, SimReady compliance
  config/            # Pipeline configuration (YAML)
tests/
data/                # Downloaded STEP files and catalog.json (gitignored)
```

### Acquisition Agent

The acquisition module (`simready/acquisition/`) automatically searches and downloads open-source STEP files. Sources are pluggable via the `@register_source` decorator pattern in `sources.py`:
- **GitHubSource** — searches public repos via GitHub code search API (set `GITHUB_TOKEN` env var for higher rate limits)
- **ABCDatasetSource** — downloads from the ABC Dataset (~1M CAD models, MIT license)

New sources implement `STEPSource.search()` and `STEPSource.download()`. Downloaded assets are tracked in `data/catalog.json`.

### Material Fidelity

Material fidelity is the primary accuracy metric. When mapping CAE materials to MDL:
- Preserve physical properties: roughness, metallic, subsurface, IOR
- Source of truth for material properties is the CAE file; do not fabricate defaults
- Validate output materials against SimReady material spec before writing USD

## Key Dependencies

| Package | Purpose |
|---|---|
| `usd-core` or `pxr` | OpenUSD Python bindings |
| `numpy` | Geometry / mesh math |
| `trimesh` | Mesh I/O and processing |
| `pythonocc-core` (OCC) | STEP/IGES CAD parsing |
| `aiohttp` | Async HTTP for acquisition agent |
| `omni.*` (Kit SDK) | Omniverse integration, MDL |

## SimReady Asset Standards

Assets must follow NVIDIA SimReady conventions:
- Correct up-axis (`Y` or `Z` depending on target)
- Meters as the default unit (`metersPerUnit = 0.01` for cm-source CAD)
- Physics schema applied where applicable (`UsdPhysics`)
- Semantic labels on prims (`SemanticsAPI`)
- MDL materials, not legacy UsdPreviewSurface, for Omniverse targets
