"""Distribution of crazyflow over multiple devices.

Sharding arrays along the world axis allows efficient parallelization of crazyflow across multiple
devices. The data itself specifies which arrays carry the world axis. See
[world_mask][crazyflow.utils.world_mask].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from crazyflow.utils import world_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import Device
    from jax.sharding import Mesh

    from crazyflow.sim.data import SimData

WORLD_AXIS = "worlds"
"""Name of the mesh axis that the worlds are distributed over."""


def world_mesh(devices: Sequence[Device]) -> Mesh:
    """Create a mesh that distributes the worlds over the devices.

    The mesh uses automatic axis types. Explicit axis types cause problems in SciPy.

    Args:
        devices: Devices to distribute the worlds over.

    Returns:
        A one-dimensional mesh over the world axis.
    """
    axes, names = (len(devices),), (WORLD_AXIS,)
    return jax.make_mesh(axes, names, axis_types=(AxisType.Auto,), devices=devices)


def placement(data: SimData, mesh: Mesh) -> SimData:
    """Build the placement that distributes the worlds of the simulation data over a mesh.

    Note:
        Pass the placement to `jax.device_put`, or modify it first to place the data by hand.

    Args:
        data: Simulation data to place.
        mesh: Mesh to distribute the worlds over, as built by
            [world_mesh][crazyflow.sim.sharding.world_mesh].

    Returns:
        A pytree of shardings matching the simulation data.
    """
    world = NamedSharding(mesh, PartitionSpec(WORLD_AXIS))
    replicated = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda indexed: world if indexed else replicated, world_mask(data))


def shard(data: SimData, mesh: Mesh) -> SimData:
    """Distribute the worlds of the simulation data over a mesh.

    Args:
        data: Simulation data to place.
        mesh: Mesh to distribute the worlds over.

    Returns:
        The placed simulation data.
    """
    return jax.device_put(data, placement(data, mesh))
