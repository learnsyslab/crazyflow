"""Utility functions for the drone models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from crazyflow._typing import Array  # To be changed to array_api_typing later


def to_xp(*args: Any, xp: ModuleType, device: Any) -> tuple[Array, ...] | Array:
    """Convert all arrays in the argument list to the given xp framework and device."""
    result = tuple(xp.asarray(x, device=device) for x in args)
    if len(result) == 1:
        return result[0]
    return result
