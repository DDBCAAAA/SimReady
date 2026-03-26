"""partnet_parser.py — Parse PartNet-Mobility URDF into a clean data model.

PartNet mobility.urdf is standard URDF with one key quirk: visual and
collision meshes share the same .obj files.  The parser maps every link to
its mesh paths and builds the full kinematic tree (joints + limits).

Output model
------------
PartNetAsset
  ├── object_id     : str
  ├── asset_dir     : Path
  ├── root_link     : str           # link with no parent joint
  ├── links         : list[PartNetLink]
  └── joints        : list[PartNetJoint]

PartNetLink
  ├── name          : str
  ├── visual_meshes : list[Path]    # absolute .obj paths
  ├── collision_meshes : list[Path] # may differ from visuals
  └── metadata      : dict          # inertial, mass if present

PartNetJoint
  ├── name          : str
  ├── joint_type    : str           # "revolute", "prismatic", "fixed", "continuous"
  ├── parent        : str           # parent link name
  ├── child         : str           # child link name
  ├── axis          : tuple[float,float,float]
  ├── origin_xyz    : tuple[float,float,float]
  ├── origin_rpy    : tuple[float,float,float]
  ├── lower         : float | None  # rad (revolute) or m (prismatic)
  └── upper         : float | None
"""
from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PartNetLink:
    name: str
    visual_meshes: list[Path] = field(default_factory=list)
    collision_meshes: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # mass, inertia, etc.


@dataclass
class PartNetJoint:
    name: str
    joint_type: str                              # urdf joint type string
    parent: str
    child: str
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower: float | None = None
    upper: float | None = None

    @property
    def is_moving(self) -> bool:
        return self.joint_type in ("revolute", "prismatic", "continuous")

    @property
    def range_rad(self) -> float | None:
        """Angular range in radians (None if limits not set)."""
        if self.lower is not None and self.upper is not None:
            return self.upper - self.lower
        return None


@dataclass
class PartNetAsset:
    object_id: str
    asset_dir: Path
    root_link: str
    links: list[PartNetLink] = field(default_factory=list)
    joints: list[PartNetJoint] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def link_map(self) -> dict[str, PartNetLink]:
        return {lnk.name: lnk for lnk in self.links}

    @property
    def joint_map(self) -> dict[str, PartNetJoint]:
        return {j.name: j for j in self.joints}

    def children_of(self, link_name: str) -> list[str]:
        """Return names of links that are direct children of link_name."""
        return [j.child for j in self.joints if j.parent == link_name]

    def moving_joints(self) -> list[PartNetJoint]:
        return [j for j in self.joints if j.is_moving]

    def all_visual_meshes(self) -> list[Path]:
        """Flat list of every visual mesh path across all links."""
        out: list[Path] = []
        for lnk in self.links:
            out.extend(lnk.visual_meshes)
        return out

    def summary(self) -> str:
        n_rev  = sum(1 for j in self.joints if j.joint_type == "revolute")
        n_pri  = sum(1 for j in self.joints if j.joint_type == "prismatic")
        n_fix  = sum(1 for j in self.joints if j.joint_type == "fixed")
        return (
            f"PartNetAsset(id={self.object_id}, root={self.root_link}, "
            f"links={len(self.links)}, joints={len(self.joints)} "
            f"[rev={n_rev}, pris={n_pri}, fixed={n_fix}])"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PartNetURDFParser:
    """Parse a PartNet-Mobility mobility.urdf into a PartNetAsset.

    Usage::

        parser = PartNetURDFParser(asset_dir=Path("data/partnet/12345"))
        asset  = parser.parse()
    """

    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = Path(asset_dir)
        self.urdf_path = self.asset_dir / "mobility.urdf"
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"mobility.urdf not found in {asset_dir}")

    def parse(self) -> PartNetAsset:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        links  = self._parse_links(root)
        joints = self._parse_joints(root)

        # Root link = link that never appears as a joint's child
        child_names = {j.child for j in joints}
        root_candidates = [lnk.name for lnk in links if lnk.name not in child_names]

        if not root_candidates:
            logger.warning(
                "No root link found (cyclic URDF?). Using first link as root."
            )
            root_link = links[0].name if links else "base"
        elif len(root_candidates) > 1:
            logger.warning(
                "Multiple root candidates %s — choosing '%s'",
                root_candidates, root_candidates[0],
            )
            root_link = root_candidates[0]
        else:
            root_link = root_candidates[0]

        object_id = self.asset_dir.name
        asset = PartNetAsset(
            object_id = object_id,
            asset_dir = self.asset_dir,
            root_link = root_link,
            links     = links,
            joints    = joints,
        )
        logger.info("Parsed %s", asset.summary())
        return asset

    # ------------------------------------------------------------------
    # Link parsing
    # ------------------------------------------------------------------

    def _parse_links(self, root: ET.Element) -> list[PartNetLink]:
        links: list[PartNetLink] = []
        for el in root.findall("link"):
            name  = el.get("name", "unnamed")
            link  = PartNetLink(name=name)

            link.visual_meshes    = self._collect_meshes(el, "visual")
            link.collision_meshes = self._collect_meshes(el, "collision")
            link.metadata         = self._parse_inertial(el)

            links.append(link)

        logger.debug("Parsed %d links", len(links))
        return links

    def _collect_meshes(self, link_el: ET.Element, tag: str) -> list[Path]:
        """Return resolved absolute paths for all <mesh> elements under <tag>."""
        paths: list[Path] = []
        for geom in link_el.findall(f"{tag}/geometry/mesh"):
            raw = geom.get("filename", "")
            if not raw:
                continue
            # URDF paths are relative to the URDF file location
            resolved = (self.asset_dir / raw).resolve()
            if not resolved.exists():
                # PartNet sometimes uses package:// URIs — strip prefix
                raw_stripped = raw.replace("package://", "").lstrip("/")
                resolved = (self.asset_dir / raw_stripped).resolve()
            if resolved.exists():
                paths.append(resolved)
            else:
                logger.debug("Mesh not found: %s (link element <%s>)", raw, tag)
        return paths

    @staticmethod
    def _parse_inertial(link_el: ET.Element) -> dict:
        """Extract mass and inertia tensor if present."""
        meta: dict = {}
        inertial = link_el.find("inertial")
        if inertial is None:
            return meta
        mass_el = inertial.find("mass")
        if mass_el is not None:
            try:
                meta["mass_kg"] = float(mass_el.get("value", 0))
            except ValueError:
                pass
        ixx = inertial.find("inertia")
        if ixx is not None:
            meta["inertia"] = {
                k: float(ixx.get(k, 0.0))
                for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            }
        return meta

    # ------------------------------------------------------------------
    # Joint parsing
    # ------------------------------------------------------------------

    def _parse_joints(self, root: ET.Element) -> list[PartNetJoint]:
        joints: list[PartNetJoint] = []
        for el in root.findall("joint"):
            name       = el.get("name", "unnamed_joint")
            joint_type = el.get("type", "fixed")

            parent = el.findtext("parent/[@link]") or ""
            parent_el = el.find("parent")
            if parent_el is not None:
                parent = parent_el.get("link", "")
            child_el = el.find("child")
            child = child_el.get("link", "") if child_el is not None else ""

            axis  = self._parse_xyz(el.find("axis"), default=(1.0, 0.0, 0.0))
            xyz, rpy = self._parse_origin(el.find("origin"))
            lower, upper = self._parse_limits(el.find("limit"), joint_type)

            joints.append(PartNetJoint(
                name       = name,
                joint_type = joint_type,
                parent     = parent,
                child      = child,
                axis       = axis,
                origin_xyz = xyz,
                origin_rpy = rpy,
                lower      = lower,
                upper      = upper,
            ))

        logger.debug("Parsed %d joints", len(joints))
        return joints

    @staticmethod
    def _parse_xyz(
        el: ET.Element | None,
        attr: str = "xyz",
        default: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float, float]:
        if el is None:
            return default
        raw = el.get(attr, "")
        parts = raw.split()
        if len(parts) == 3:
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                pass
        return default

    @staticmethod
    def _parse_origin(
        el: ET.Element | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return (xyz_tuple, rpy_tuple) from a URDF <origin> element."""
        zero3 = (0.0, 0.0, 0.0)
        if el is None:
            return zero3, zero3

        def _parse3(attr: str) -> tuple[float, float, float]:
            raw = el.get(attr, "")
            parts = raw.split()
            if len(parts) == 3:
                try:
                    return (float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
            return zero3

        return _parse3("xyz"), _parse3("rpy")

    @staticmethod
    def _parse_limits(
        el: ET.Element | None,
        joint_type: str,
    ) -> tuple[float | None, float | None]:
        """Parse <limit lower=... upper=...>."""
        if el is None or joint_type in ("fixed", "floating"):
            return None, None
        try:
            lower = float(el.get("lower", math.nan))
            upper = float(el.get("upper", math.nan))
            return (
                lower if not math.isnan(lower) else None,
                upper if not math.isnan(upper) else None,
            )
        except ValueError:
            return None, None
