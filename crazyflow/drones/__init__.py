"""Hardware descriptions for the supported drone platforms.

This package bundles the MuJoCo MJCF scene files that define each drone configuration and their
referenced meshes (``assets/``). For the physical params, see [crazyflow.dynamics.load_params][].

Use ``available_drones`` to enumerate the supported configurations.
"""

from pathlib import Path

__all__ = ["available_drones"]

_mjcf_files = sorted(Path(__file__).parent.glob("*.xml"))
available_drones: tuple[str, ...] = tuple(p.stem for p in _mjcf_files)
"""Names of all drone configurations, i.e. the MJCF files in ``crazyflow/drones``."""
