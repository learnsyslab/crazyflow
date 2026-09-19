"""Hardware descriptions for the supported drone platforms.

This package bundles the MuJoCo MJCF scene files that define each drone configuration and their
referenced meshes (``assets/``). For the physical params, see [crazyflow.dynamics.load_params][].

Use ``Drone`` to enumerate the supported configurations.
"""

from enum import StrEnum
from pathlib import Path

__all__ = ["Drone"]


class Drone(StrEnum):
    """Drone configurations. Each member has an MJCF file ``crazyflow/drones/<name>.xml``."""

    cf21B_500 = "cf21B_500"
    cf2x_L250 = "cf2x_L250"
    cf2x_P250 = "cf2x_P250"
    cf2x_T350 = "cf2x_T350"


# Sanity check at startup
_mjcf_files = {p.stem for p in Path(__file__).parent.glob("*.xml")}
assert {d.value for d in Drone} == _mjcf_files, (
    f"Drone enum {sorted(d.value for d in Drone)} does not match MJCF files {sorted(_mjcf_files)}"
)
