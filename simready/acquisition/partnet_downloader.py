"""partnet_downloader.py — Fetch PartNet-Mobility objects by ID.

PartNet-Mobility is mirrored on HuggingFace (haosulab/PartNet-Mobility).
Each object is a zip archive with the canonical directory layout:

    <object_id>/
        mobility.urdf
        meta.json
        textured_objs/
            <part>.obj
            <part>.mtl
            ...

Primary source  : HuggingFace Hub  (no auth needed for public repo)
Fallback source : SAPIEN CDN       (requires HF token for gated repos)

Install extras:
    pip install huggingface_hub requests tqdm
"""
from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID      = "haosulab/PartNet-Mobility"
HF_REPO_TYPE    = "dataset"
HF_FILE_PATTERN = "data/{object_id}.zip"           # path inside the HF repo

SAPIEN_CDN_URL  = (
    "https://sapien.cs.columbia.edu/api/download/{object_id}.zip"
)

_REQUIRED_ENTRIES = {"mobility.urdf"}               # zip must contain these


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    """Outcome of a single object download attempt."""
    object_id: str
    success: bool
    asset_dir: Path | None = None       # extracted root directory
    error: str | None = None


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class PartNetDownloader:
    """Downloads and extracts PartNet-Mobility objects.

    Parameters
    ----------
    dest_dir:
        Root directory under which each object gets its own sub-folder,
        e.g.  dest_dir/12345/mobility.urdf
    hf_token:
        HuggingFace API token.  Only required if the repo is gated.
        Pass None for the public mirror (default).
    force_redownload:
        Re-download even if the object directory already exists locally.
    """

    def __init__(
        self,
        dest_dir: Path,
        hf_token: str | None = None,
        force_redownload: bool = False,
    ) -> None:
        self.dest_dir        = Path(dest_dir)
        self.hf_token        = hf_token
        self.force_redownload = force_redownload
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, object_id: str | int) -> DownloadResult:
        """Download and extract one PartNet-Mobility object.

        Returns a :class:`DownloadResult` regardless of success.
        Callers should check ``.success`` before proceeding.
        """
        oid       = str(object_id)
        asset_dir = self.dest_dir / oid

        if asset_dir.exists() and not self.force_redownload:
            urdf = asset_dir / "mobility.urdf"
            if urdf.exists():
                logger.info("Object %s already present at %s", oid, asset_dir)
                return DownloadResult(object_id=oid, success=True, asset_dir=asset_dir)
            # Corrupted / partial — blow away and re-fetch
            shutil.rmtree(asset_dir, ignore_errors=True)

        zip_path = self.dest_dir / f"{oid}.zip"

        try:
            self._download(oid, zip_path)
            self._extract(zip_path, asset_dir, oid)
            zip_path.unlink(missing_ok=True)
            logger.info("Object %s ready at %s", oid, asset_dir)
            return DownloadResult(object_id=oid, success=True, asset_dir=asset_dir)

        except Exception as exc:
            zip_path.unlink(missing_ok=True)
            logger.error("Failed to fetch object %s: %s", oid, exc)
            return DownloadResult(object_id=oid, success=False, error=str(exc))

    def fetch_batch(
        self,
        object_ids: list[str | int],
    ) -> list[DownloadResult]:
        """Download multiple objects sequentially, collecting all results."""
        results: list[DownloadResult] = []
        for oid in object_ids:
            results.append(self.fetch(oid))
        passed  = sum(1 for r in results if r.success)
        failed  = len(results) - passed
        logger.info("Batch complete: %d/%d downloaded (%d failed)", passed, len(results), failed)
        return results

    # ------------------------------------------------------------------
    # Download strategies (tried in order)
    # ------------------------------------------------------------------

    def _download(self, oid: str, dest_zip: Path) -> None:
        """Try HF Hub first, then direct HTTP CDN."""
        try:
            self._download_hf(oid, dest_zip)
        except Exception as hf_exc:
            logger.debug("HF download failed (%s) — trying CDN fallback", hf_exc)
            self._download_http(oid, dest_zip)

    def _download_hf(self, oid: str, dest_zip: Path) -> None:
        """Pull the zip from HuggingFace Hub (handles auth + caching)."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        hf_path = HF_FILE_PATTERN.format(object_id=oid)
        logger.info("HF: downloading %s from %s/%s", hf_path, HF_REPO_ID, hf_path)
        local = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = hf_path,
            repo_type = HF_REPO_TYPE,
            token     = self.hf_token,
            local_dir = str(self.dest_dir),
        )
        shutil.copy2(local, dest_zip)

    def _download_http(self, oid: str, dest_zip: Path) -> None:
        """Stream the zip directly from the SAPIEN CDN."""
        url = SAPIEN_CDN_URL.format(object_id=oid)
        logger.info("CDN: downloading %s from %s", oid, url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0))
        with dest_zip.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"PartNet {oid}", leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract(self, zip_path: Path, asset_dir: Path, oid: str) -> None:
        """Extract zip into asset_dir, normalising the internal layout.

        PartNet zips may be rooted under <object_id>/ or flat.  We always
        produce:   asset_dir/mobility.urdf   (no extra level of nesting).
        """
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            self._validate_zip(names, oid)

            # Detect common root prefix (e.g. "12345/" or "")
            prefix = _common_zip_prefix(names, oid)
            asset_dir.mkdir(parents=True, exist_ok=True)

            for member in names:
                if member.endswith("/"):
                    continue            # directory entry
                rel = member[len(prefix):]  # strip common prefix
                if not rel:
                    continue
                dest = asset_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        logger.debug("Extracted %d files to %s", len(names), asset_dir)

    @staticmethod
    def _validate_zip(names: list[str], oid: str) -> None:
        flat = {n.split("/")[-1] for n in names}
        missing = _REQUIRED_ENTRIES - flat
        if missing:
            raise ValueError(
                f"Object {oid}: zip is missing required files: {missing}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _common_zip_prefix(names: list[str], oid: str) -> str:
    """Return the common leading path to strip (e.g. '12345/'), or ''."""
    candidates = {n.split("/")[0] for n in names if "/" in n}
    if len(candidates) == 1:
        prefix = candidates.pop() + "/"
        # Only strip if every name starts with this prefix
        if all(n.startswith(prefix) or not n for n in names):
            return prefix
    return ""
