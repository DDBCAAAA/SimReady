"""SLDPRT → STEP converter.

Two-path strategy:
  Path A — FreeCAD (no API key, headless subprocess):
    Works for SolidWorks 2018 and earlier (OLE2/CFB container format).

  Path B — Autodesk APS Model Derivative API (requires API keys):
    Works for ALL SLDPRT versions including SolidWorks 2019+.
    Needs env vars: AUTODESK_CLIENT_ID and AUTODESK_CLIENT_SECRET.
    SLDPRT is uploaded → converted to OBJ → downloaded → written as
    STEP AP214 (tessellated B-Shell) via OCC.

Usage:
    from simready.ingestion.sldprt_converter import convert_sldprt
    step_path = convert_sldprt("part.sldprt", "part.step")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# OLE2/CFB magic (SolidWorks 2018 and earlier)
_CFB_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")

# Autodesk APS base URLs
_APS_AUTH_URL  = "https://developer.api.autodesk.com/authentication/v2/token"
_APS_OSS_BASE  = "https://developer.api.autodesk.com/oss/v2/buckets"
_APS_MD_BASE   = "https://developer.api.autodesk.com/modelderivative/v2/designdata"

FREECAD_PYTHON = "/Applications/FreeCAD.app/Contents/Resources/bin/python"
FREECAD_LIB    = "/Applications/FreeCAD.app/Contents/Resources/lib"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_old_format(path: Path) -> bool:
    """Return True if the SLDPRT uses the pre-2019 OLE2/CFB container."""
    with open(path, "rb") as fh:
        return fh.read(8) == _CFB_MAGIC


# ---------------------------------------------------------------------------
# Path A — FreeCAD
# ---------------------------------------------------------------------------

_FREECAD_SCRIPT = """\
import sys
sys.path.insert(0, {lib!r})
import FreeCAD, Part

doc = FreeCAD.newDocument()
shape = Part.read({src!r})
Part.export([shape], {dst!r})
FreeCAD.closeDocument(doc.Name)
print("OK")
"""


def _convert_via_freecad(src: Path, dst: Path) -> None:
    """Convert an old-format SLDPRT to STEP using FreeCAD headlessly."""
    if not Path(FREECAD_PYTHON).exists():
        raise RuntimeError(
            "FreeCAD not found at expected path. "
            "Install with: brew install --cask freecad"
        )

    script = _FREECAD_SCRIPT.format(
        lib=FREECAD_LIB,
        src=str(src.resolve()),
        dst=str(dst.resolve()),
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        script_path = tf.name

    try:
        result = subprocess.run(
            [FREECAD_PYTHON, script_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or "OK" not in result.stdout:
            raise RuntimeError(
                f"FreeCAD conversion failed:\n{result.stderr.strip()}"
            )
    finally:
        Path(script_path).unlink(missing_ok=True)

    if not dst.exists() or dst.stat().st_size < 100:
        raise RuntimeError("FreeCAD produced an empty output file.")

    logger.info("FreeCAD: converted %s → %s", src.name, dst.name)


# ---------------------------------------------------------------------------
# Path B — Autodesk APS
# ---------------------------------------------------------------------------

def _aps_token(client_id: str, client_secret: str) -> str:
    """Obtain a 2-legged Bearer token from APS."""
    import urllib.request, urllib.parse
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "data:read data:write bucket:read bucket:create",
    }).encode()
    req = urllib.request.Request(
        _APS_AUTH_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {creds}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["access_token"]


def _aps_request(method: str, url: str, token: str,
                 body: bytes | None = None,
                 content_type: str = "application/json",
                 extra_headers: dict | None = None) -> dict | bytes:
    """Generic APS REST call. Returns parsed JSON or raw bytes."""
    import urllib.request
    headers: dict = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return json.loads(raw)
            return raw
    except Exception as exc:
        raise RuntimeError(f"APS request failed [{method} {url}]: {exc}") from exc


def _aps_upload(bucket: str, obj_key: str, file_path: Path, token: str) -> str:
    """Upload a file to APS OSS. Returns the encoded URN."""
    import urllib.request

    # 1. Get signed upload URL
    url = f"{_APS_OSS_BASE}/{bucket}/objects/{obj_key}/signeds3upload?parts=1"
    upload_info = _aps_request("GET", url, token)
    signed_url  = upload_info["urls"][0]
    upload_key  = upload_info["uploadKey"]

    # 2. PUT file bytes to S3
    file_bytes = file_path.read_bytes()
    put_req = urllib.request.Request(signed_url, data=file_bytes, method="PUT")
    with urllib.request.urlopen(put_req, timeout=300) as resp:
        etag = resp.headers.get("ETag", "").strip('"')

    # 3. Finalise upload
    fin_url  = f"{_APS_OSS_BASE}/{bucket}/objects/{obj_key}/signeds3upload"
    fin_body = json.dumps({"uploadKey": upload_key, "parts": [{"partNumber": 1, "etag": etag}]}).encode()
    _aps_request("POST", fin_url, token, body=fin_body)

    # URN = base64-url-safe of "urn:adsk.objects:os.object:{bucket}/{key}"
    raw_urn = f"urn:adsk.objects:os.object:{bucket}/{obj_key}"
    return base64.urlsafe_b64encode(raw_urn.encode()).decode().rstrip("=")


def _aps_translate(urn: str, token: str) -> None:
    """Submit a Model Derivative translation job (SLDPRT → OBJ)."""
    body = json.dumps({
        "input":  {"urn": urn},
        "output": {
            "destination": {"region": "us"},
            "formats": [{"type": "obj"}],
        },
    }).encode()
    _aps_request("POST", f"{_APS_MD_BASE}/job", token,
                 body=body, extra_headers={"x-ads-force": "true"})


def _aps_wait(urn: str, token: str, poll_interval: float = 5.0, timeout: float = 600.0) -> dict:
    """Poll the manifest until status == 'success' or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        manifest = _aps_request("GET", f"{_APS_MD_BASE}/{urn}/manifest", token)
        status = manifest.get("status", "")
        logger.info("APS translation status: %s", status)
        if status == "success":
            return manifest
        if status in ("failed", "timeout"):
            msgs = manifest.get("derivatives", [{}])[0].get("messages", [])
            raise RuntimeError(f"APS translation {status}: {msgs}")
        time.sleep(poll_interval)
    raise TimeoutError(f"APS translation did not complete within {timeout}s")


def _aps_download_obj(manifest: dict, urn: str, token: str) -> bytes:
    """Download the first OBJ derivative from a completed manifest."""
    for deriv in manifest.get("derivatives", []):
        if deriv.get("outputType") == "obj":
            for child in deriv.get("children", []):
                child_urn = child.get("urn", "")
                if child_urn.lower().endswith(".obj"):
                    url = f"{_APS_MD_BASE}/{urn}/manifest/{child_urn}"
                    data = _aps_request("GET", url, token)
                    if isinstance(data, (bytes, bytearray)):
                        return data
    raise RuntimeError("No OBJ derivative found in APS manifest.")


def _convert_via_aps(src: Path, dst: Path,
                     client_id: str, client_secret: str) -> None:
    """Full APS pipeline: upload → translate → download OBJ → write STEP."""
    # Use a transient bucket (auto-deleted after 24 h)
    bucket = f"simready-{uuid.uuid4().hex[:12]}"
    obj_key = src.name.replace(" ", "_")

    logger.info("APS: authenticating …")
    token = _aps_token(client_id, client_secret)

    # Create transient bucket (ignore 409 = already exists)
    try:
        body = json.dumps({"bucketKey": bucket, "policyKey": "transient"}).encode()
        _aps_request("POST", _APS_OSS_BASE, token, body=body)
    except RuntimeError as exc:
        if "409" not in str(exc):
            raise

    logger.info("APS: uploading %s (%d KB) …", src.name, src.stat().st_size // 1024)
    urn = _aps_upload(bucket, obj_key, src, token)

    logger.info("APS: submitting translation job …")
    _aps_translate(urn, token)

    logger.info("APS: waiting for translation (up to 10 min) …")
    manifest = _aps_wait(urn, token)

    logger.info("APS: downloading OBJ derivative …")
    obj_bytes = _aps_download_obj(manifest, urn, token)

    logger.info("APS: converting OBJ mesh → STEP …")
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tf:
        tf.write(obj_bytes)
        obj_path = Path(tf.name)

    try:
        _obj_to_step(obj_path, dst)
    finally:
        obj_path.unlink(missing_ok=True)

    logger.info("APS: wrote %s (%.1f KB)", dst.name, dst.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Mesh (OBJ/trimesh) → STEP via OCC
# ---------------------------------------------------------------------------

def _obj_to_step(obj_path: Path, step_path: Path) -> None:
    """Load an OBJ with trimesh, build OCC compound, export as STEP AP214."""
    import trimesh
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static
    from OCP.IFSelect import IFSelect_RetDone

    scene = trimesh.load(str(obj_path), force="mesh", process=False)
    meshes = (
        list(scene.geometry.values())
        if isinstance(scene, trimesh.Scene)
        else [scene]
    )

    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    total_faces = 0
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        verts = mesh.vertices
        for tri in mesh.faces:
            p0 = gp_Pnt(*map(float, verts[tri[0]]))
            p1 = gp_Pnt(*map(float, verts[tri[1]]))
            p2 = gp_Pnt(*map(float, verts[tri[2]]))
            poly = BRepBuilderAPI_MakePolygon(p0, p1, p2, True)
            if not poly.IsDone():
                continue
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if face.IsDone():
                builder.Add(compound, face.Shape())
                total_faces += 1

    if total_faces == 0:
        raise RuntimeError("No valid triangular faces could be built from the OBJ mesh.")

    Interface_Static.SetCVal("write.step.schema", "AP214")
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError("OCC STEP writer failed.")

    logger.info("mesh→STEP: %d triangular faces written", total_faces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_sldprt(
    sldprt_path: str | Path,
    step_path: str | Path,
    *,
    autodesk_client_id: str | None = None,
    autodesk_client_secret: str | None = None,
) -> Path:
    """Convert a SolidWorks SLDPRT file to STEP.

    Path A (no API key):  FreeCAD headless — works for SolidWorks ≤ 2018.
    Path B (API key):     Autodesk APS cloud — works for all SLDPRT versions.

    API keys can be supplied as arguments or via environment variables
    AUTODESK_CLIENT_ID / AUTODESK_CLIENT_SECRET (free APS dev account).

    Args:
        sldprt_path:  Input .sldprt file.
        step_path:    Desired output .step file.
        autodesk_client_id:     APS Client ID (optional).
        autodesk_client_secret: APS Client Secret (optional).

    Returns:
        Path to the written STEP file.

    Raises:
        RuntimeError if both conversion paths fail.
    """
    src = Path(sldprt_path)
    dst = Path(step_path)
    if not src.exists():
        raise FileNotFoundError(f"SLDPRT not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    client_id     = autodesk_client_id     or os.environ.get("AUTODESK_CLIENT_ID", "")
    client_secret = autodesk_client_secret or os.environ.get("AUTODESK_CLIENT_SECRET", "")

    # Path A: try FreeCAD for old-format files
    if _is_old_format(src):
        logger.info("SLDPRT format: OLE2/CFB (≤2018) — trying FreeCAD")
        try:
            _convert_via_freecad(src, dst)
            return dst
        except Exception as exc:
            logger.warning("FreeCAD path failed: %s", exc)
            if not client_id:
                raise RuntimeError(
                    f"FreeCAD conversion failed and no APS credentials available.\n"
                    f"Set AUTODESK_CLIENT_ID / AUTODESK_CLIENT_SECRET or ensure "
                    f"FreeCAD is installed at {FREECAD_PYTHON}"
                ) from exc
    else:
        logger.info("SLDPRT format: SolidWorks 2019+ (new binary format) — requires APS")
        if not client_id:
            raise RuntimeError(
                "SolidWorks 2019+ SLDPRT files cannot be converted by FreeCAD.\n"
                "Provide Autodesk APS credentials via:\n"
                "  env vars:  AUTODESK_CLIENT_ID, AUTODESK_CLIENT_SECRET\n"
                "  or args:   convert_sldprt(..., autodesk_client_id=..., autodesk_client_secret=...)\n"
                "Sign up free at https://aps.autodesk.com/"
            )

    # Path B: Autodesk APS cloud conversion
    logger.info("Using Autodesk APS cloud conversion")
    _convert_via_aps(src, dst, client_id, client_secret)
    return dst
