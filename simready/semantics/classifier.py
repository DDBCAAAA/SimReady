"""Semantic classifier — maps part names to SimReady taxonomy labels.

Taxonomy follows NVIDIA SimReady conventions:
  <category>:<subcategory>

Categories:
  fastener          — bolts, nuts, screws, washers, rivets, pins
  mechanical        — gears, bearings, shafts, cams, springs, pulleys
  structural        — beams, plates, brackets, frames, flanges, enclosures
  fluid_system      — pipes, valves, fittings, nozzles, manifolds
  electrical        — connectors, housings, terminals, PCBs
  industrial_part   — generic manufactured component (fallback)
"""

from __future__ import annotations

_TAXONOMY: list[tuple[str, list[str]]] = [
    # (label, [keywords])
    ("fastener:bolt",       ["bolt", "screw", "stud"]),
    ("fastener:nut",        ["nut", "hexnut", "locknut"]),
    ("fastener:washer",     ["washer"]),
    ("fastener:rivet",      ["rivet"]),
    ("fastener:pin",        ["pin", "dowel", "clevis"]),
    ("mechanical:gear",     ["gear", "pinion", "sprocket"]),
    ("mechanical:bearing",  ["bearing", "race", "roller"]),
    ("mechanical:shaft",    ["shaft", "axle", "spindle"]),
    ("mechanical:spring",   ["spring", "coil"]),
    ("mechanical:pulley",   ["pulley", "sheave"]),
    ("mechanical:cam",      ["cam", "eccentric"]),
    ("structural:plate",    ["plate", "panel", "slab"]),
    ("structural:bracket",  ["bracket", "mount", "clamp", "clip"]),
    ("structural:beam",     ["beam", "bar", "rod", "rail"]),
    ("structural:frame",    ["frame", "chassis", "skeleton"]),
    ("structural:flange",   ["flange", "collar", "boss"]),
    ("structural:enclosure", ["enclosure", "box", "housing", "cover", "shell", "lid"]),
    ("fluid_system:pipe",   ["pipe", "tube", "duct", "hose"]),
    ("fluid_system:valve",  ["valve", "cock", "gate"]),
    ("fluid_system:fitting", ["fitting", "coupling", "elbow", "tee", "union"]),
    ("fluid_system:nozzle", ["nozzle", "orifice", "injector"]),
    ("electrical:connector", ["connector", "plug", "socket", "jack"]),
    ("electrical:housing",  ["terminal", "pcb", "circuit"]),
]


def classify(name: str) -> str:
    """Return a SimReady taxonomy label for a part name.

    Args:
        name: Part name (e.g. "gear_spur", "PLATE", "bolt_m8").

    Returns:
        Taxonomy string like "mechanical:gear" or "industrial_part:component".
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for label, keywords in _TAXONOMY:
        if any(kw in name_lower for kw in keywords):
            return label
    return "industrial_part:component"
