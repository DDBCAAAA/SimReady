"""STEP file reader — parses .step/.stp CAD files into an intermediate geometry representation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CADBody:
    """A single solid body extracted from a CAD file."""

    name: str
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int — triangle indices
    normals: np.ndarray | None = None  # (N, 3) float64
    material_name: str | None = None
    metadata: dict = field(default_factory=dict)
    # LOD meshes: list of (ratio, vertices, faces) generated during processing.
    # Index 0 = full detail (ratio 1.0), index 1+ = decimated.
    lod_meshes: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)


@dataclass
class CADAssembly:
    """Top-level container for an imported CAD file."""

    source_path: Path
    bodies: list[CADBody] = field(default_factory=list)
    units: str = "mm"  # source file units
    metadata: dict = field(default_factory=dict)


def _detect_step_units(path: Path) -> str:
    """Detect the length unit by parsing the first 100 lines of a STEP file.

    Looks for standard STEP unit declarations:
      - ``SI_UNIT(.MILLI.,.METRE.)``     → 'mm'
      - ``SI_UNIT($,.METRE.)``           → 'm'
      - ``SI_UNIT(.CENTI.,.METRE.)``     → 'cm'
      - ``CONVERSION_BASED_UNIT('INCH'`` → 'in'

    Falls back to 'mm' with a warning when no declaration is found (most
    CAD software exports in millimeters by default).

    Returns:
        Unit string: ``'mm'``, ``'m'``, ``'cm'``, or ``'in'``.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(100)).upper()
    except Exception as exc:
        logger.warning(
            "Could not read STEP header of %s (%s). Defaulting to millimeters.",
            path.name, exc,
        )
        return "mm"

    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", head):
        return "mm"
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", head):
        return "m"
    if re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", head):
        return "cm"
    if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH", head):
        return "in"

    logger.warning(
        "No unit metadata found in first 100 lines of %s. "
        "Defaulting to millimeters (0.001 scale).",
        path.name,
    )
    return "mm"


def read_step(path: Path, tessellation_tolerance: float = 0.001) -> CADAssembly:
    """Read a STEP file and tessellate BRep geometry into triangle meshes.

    Args:
        path: Path to the .step or .stp file.
        tessellation_tolerance: Chord tolerance for BRep → mesh conversion (meters).

    Returns:
        CADAssembly containing tessellated bodies.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".step", ".stp"):
        raise ValueError(f"Expected .step or .stp file, got: {suffix}")

    try:
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        from OCP.IFSelect import IFSelect_RetDone
        # OCP 7.7 used `topods` alias; 7.8+ uses TopoDS with static _s methods
        try:
            from OCP.TopoDS import topods as _topods
            _solid = _topods.Solid
            _face  = _topods.Face
        except ImportError:
            from OCP.TopoDS import TopoDS as _topods_cls
            _solid = _topods_cls.Solid_s
            _face  = _topods_cls.Face_s
    except ImportError:
        raise ImportError(
            "OCP (pythonocc-core) is required for STEP reading. "
            "Install with: pip install -e '.[cad]'"
        )

    # --- Detect source units before tessellation ---
    detected_unit = _detect_step_units(path)
    logger.info("Detected source unit '%s' in %s", detected_unit, path.name)

    # --- Try XDE reader first (preserves PRODUCT names and hierarchy) ---
    shape, product_names = _read_step_xde(path, tessellation_tolerance)

    # Count solids for progress display
    _counter = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while _counter.More():
        solid_count += 1
        _counter.Next()
    logger.info("Found %d solid bodies in %s — tessellating...", solid_count, path.name)

    try:
        from tqdm import tqdm as _tqdm
        _pbar = _tqdm(total=solid_count, desc=f"Tessellating {path.stem}", unit="body",
                      dynamic_ncols=True)
    except ImportError:
        _pbar = None

    assembly = CADAssembly(source_path=path, units=detected_unit)
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    body_idx = 0

    while explorer.More():
        # Capture assembly-level location BEFORE downcast — _solid() preserves the
        # type but the Location on the original explorer shape is what we need.
        solid_shape = explorer.Current()
        solid_location = solid_shape.Location()
        solid = _solid(solid_shape)
        mesh = BRepMesh_IncrementalMesh(solid, tessellation_tolerance)
        mesh.Perform()

        if not mesh.IsDone():
            logger.warning("Tessellation failed for body %d, skipping", body_idx)
            if _pbar is not None:
                _pbar.update(1)
            explorer.Next()
            body_idx += 1
            continue

        # Extract triangulated faces from the solid
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            face = _face(face_explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)

            if triangulation is None:
                face_explorer.Next()
                continue

            n_nodes = triangulation.NbNodes()
            n_tris = triangulation.NbTriangles()

            # Extract vertices — BRep_Tool.Triangulation_s sets `location` to the
            # face's placement within the solid (solid-local coords after transform).
            verts = np.empty((n_nodes, 3), dtype=np.float64)
            for i in range(1, n_nodes + 1):
                pnt = triangulation.Node(i)
                pnt_transformed = pnt.Transformed(location.Transformation())
                verts[i - 1] = [pnt_transformed.X(), pnt_transformed.Y(), pnt_transformed.Z()]
            all_vertices.append(verts)

            # Extract triangles
            tris = np.empty((n_tris, 3), dtype=np.int64)
            for i in range(1, n_tris + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                tris[i - 1] = [n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset]
            all_faces.append(tris)

            vertex_offset += n_nodes
            face_explorer.Next()

        if all_vertices:
            verts_array = np.vstack(all_vertices)

            # Bake the solid's assembly-level transform into vertices so every body
            # lives in world space.  Without this, all bodies cluster at (0,0,0)
            # because BRep_Tool.Triangulation_s returns solid-local coordinates.
            if not solid_location.IsIdentity():
                trsf = solid_location.Transformation()
                # OCC 3×4 matrix: Value(row 1-3, col 1-4); col 4 = translation.
                rot = np.array(
                    [[trsf.Value(r, c) for c in range(1, 4)] for r in range(1, 4)],
                    dtype=np.float64,
                )
                trans = np.array([trsf.Value(r, 4) for r in range(1, 4)], dtype=np.float64)
                verts_array = verts_array @ rot.T + trans

            # Use extracted product name when available; fall back to body index
            body_name = product_names.get(body_idx, f"body_{body_idx}")
            print(f"Part: {body_name}, World Centroid: {verts_array.mean(axis=0)}")
            if _pbar is not None:
                _pbar.set_postfix(name=body_name[:20], verts=len(verts_array))
            body = CADBody(
                name=body_name,
                vertices=verts_array,
                faces=np.vstack(all_faces),
                material_name=body_name,
            )
            assembly.bodies.append(body)

        if _pbar is not None:
            _pbar.update(1)
        explorer.Next()
        body_idx += 1

    if _pbar is not None:
        _pbar.close()

    logger.info("Read %d bodies from %s", len(assembly.bodies), path.name)
    return assembly


def _read_step_xde(path: Path, tessellation_tolerance: float):
    """Read a STEP file via STEPCAFControl_Reader (XDE) to preserve product names.

    Returns:
        (shape, product_names): the merged OCC shape and a dict {body_idx: name}.
        Falls back to STEPControl_Reader if XDE is unavailable or fails.
    """
    try:
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.TDocStd import TDocStd_Document
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID

        doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
        caf_reader = STEPCAFControl_Reader()
        caf_reader.SetNameMode(True)
        status = caf_reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"XDE ReadFile failed: {status}")
        caf_reader.Transfer(doc)

        # Extract product names from XDE label tree
        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        product_names: dict[int, str] = {}
        _collect_product_names(shape_tool, product_names)

        # Merge all free shapes into a single compound
        free_labels = []
        shape_tool.GetFreeShapes(free_labels)
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for label in free_labels:
            s = shape_tool.GetShape_s(label)
            builder.Add(compound, s)

        logger.info(
            "XDE extracted %d product name(s) from %s", len(product_names), path.name
        )
        return compound, product_names

    except Exception as exc:
        logger.debug("XDE extraction failed (%s), falling back to STEPControl_Reader", exc)
        return _read_step_basic(path)


def _collect_product_names(shape_tool, out: dict, _seen=None) -> None:
    """Walk the XDE shape tree and collect {solid_index: product_name} mappings."""
    from OCP.XCAFDoc import XCAFDoc_Name
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    if _seen is None:
        _seen = set()

    labels = []
    shape_tool.GetFreeShapes(labels)
    solid_idx = 0

    def _name_from_label(label):
        attr = XCAFDoc_Name()
        if label.FindAttribute(XCAFDoc_Name.GetID_s(), attr):
            return attr.Get().ToExtString().strip()
        return None

    def _walk(label, depth=0):
        nonlocal solid_idx
        shape = shape_tool.GetShape_s(label)
        label_id = label.EntryDumpToString()
        if label_id in _seen:
            return
        _seen.add(label_id)

        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            name = _name_from_label(label)
            if name:
                out[solid_idx] = name
            solid_idx += 1

        # Recurse into components
        components = []
        shape_tool.GetComponents_s(label, components)
        for comp in components:
            ref_label = comp
            if shape_tool.IsReference_s(comp):
                shape_tool.GetReferredShape_s(comp, ref_label)
            _walk(ref_label, depth + 1)

    for lbl in labels:
        _walk(lbl)


def _read_step_basic(path: Path):
    """Minimal fallback reader using STEPControl_Reader (no names)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path} (status={status})")
    reader.TransferRoots()
    return reader.OneShape(), {}
