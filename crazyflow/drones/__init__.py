"""Hardware descriptions for the supported drone platforms.

This package bundles the MuJoCo MJCF scene files that define each drone configuration and their
referenced meshes (``assets/``). For the physical params, see [crazyflow.dynamics.load_params][].

Use ``Drone`` to enumerate the supported configurations.
"""

from enum import StrEnum
from pathlib import Path

__all__ = ["Drone"]

_drones = [p.stem for p in sorted(Path(__file__).parent.glob("*.xml"))]
Drone: StrEnum = StrEnum("Drone", [(name, name) for name in _drones])
"""Drone configurations, i.e. the MJCF files in ``crazyflow/drones``."""
