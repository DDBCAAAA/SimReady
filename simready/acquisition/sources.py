"""Base interface and registry for STEP file sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class STEPAsset:
    """Metadata for a discovered STEP file."""

    name: str
    url: str
    source: str  # e.g. "github", "abc_dataset"
    size_bytes: int | None = None
    license: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    local_path: Path | None = None  # set after download
    usd_path: Path | None = None  # set after USD conversion


class STEPSource(abc.ABC):
    """Base class for STEP file sources."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[STEPAsset]:
        """Search for STEP files matching a query."""
        ...

    @abc.abstractmethod
    async def download(self, asset: STEPAsset, dest_dir: Path) -> Path:
        """Download a STEP file to dest_dir. Returns the local file path."""
        ...


_REGISTRY: dict[str, type[STEPSource]] = {}


def register_source(cls: type[STEPSource]) -> type[STEPSource]:
    """Decorator to register a STEP source."""
    _REGISTRY[cls.name.fget(None)] = cls  # type: ignore[attr-defined]
    return cls


def get_source(name: str) -> STEPSource:
    """Instantiate a registered source by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown source '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_sources() -> list[str]:
    """Return names of all registered sources."""
    return list(_REGISTRY.keys())
