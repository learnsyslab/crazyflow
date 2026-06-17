"""Utility functions for safer access to OrderedDicts.

We use OrderedDicts to store the functions of the step and reset pipelines. This preserves the order
of stages. To allow users to modify the pipelines by names, safely adding elements without
overwriting existing ones, and to improve readibility, we add utility functions for adding,
replacing, and inserting stages.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def append_fn(pipeline: OrderedDict[str, F], fn: F, name: str | None = None):
    """Append a function to the end of the pipeline.

    Args:
        pipeline: The pipeline to modify.
        fn: The function to add.
        name: Unique name of the new stage. Defaults to ``fn.__name__``.

    Raises:
        KeyError: If a stage with the same name already exists.
    """
    pipeline[_resolve_name(pipeline, name, fn)] = fn


def prepend_fn(pipeline: OrderedDict[str, F], fn: F, name: str | None = None):
    """Prepend a function to the beginning of the pipeline.

    Args:
        pipeline: The pipeline to modify.
        fn: The function to add.
        name: Unique name of the new stage. Defaults to ``fn.__name__``.

    Raises:
        KeyError: If a stage with the same name already exists.
    """
    items = list(pipeline.items())
    items.insert(0, (_resolve_name(pipeline, name, fn), fn))
    pipeline.clear()
    pipeline.update(OrderedDict(items))


def replace_fn(pipeline: OrderedDict[str, F], fn: F, name: str):
    """Replace the function of an existing stage in the pipeline.

    Args:
        pipeline: The pipeline to modify.
        fn: The new function for the stage.
        name: Name of the stage to replace.

    Raises:
        KeyError: If no stage with this name exists.
    """
    if name not in pipeline:
        raise KeyError(f"No stage named '{name}'. Available stages: {tuple(pipeline.keys())}")
    pipeline[name] = fn


def remove_fn(pipeline: OrderedDict[str, F], name: str):
    """Remove a stage from the pipeline.

    Args:
        pipeline: The pipeline to modify.
        name: Name of the stage to remove.
    """
    if name not in pipeline:
        raise KeyError(f"No stage named '{name}'. Available stages: {tuple(pipeline.keys())}")
    del pipeline[name]


def insert_fn_before(pipeline: OrderedDict[str, F], anchor: str, fn: F, name: str | None = None):
    """Insert a function directly before an existing stage in the pipeline.

    Args:
        pipeline: The pipeline to modify.
        anchor: Name of an existing stage to insert before.
        fn: The function to insert.
        name: Unique name of the new stage. Defaults to ``fn.__name__``.
    """
    _insert(pipeline, anchor, fn, _resolve_name(pipeline, name, fn), offset=0)


def insert_fn_after(pipeline: OrderedDict[str, F], anchor: str, fn: F, name: str | None = None):
    """Insert a function directly after an existing stage in the pipeline.

    Args:
        pipeline: The pipeline to modify.
        anchor: Name of an existing stage to insert after.
        fn: The function to insert.
        name: Unique name of the new stage. Defaults to ``fn.__name__``.
    """
    _insert(pipeline, anchor, fn, _resolve_name(pipeline, name, fn), offset=1)


def _insert(pipeline: OrderedDict[str, F], anchor: str, fn: F, name: str | None, offset: int):
    if anchor not in pipeline:
        raise KeyError(f"No stage named '{anchor}'. Available stages: {tuple(pipeline.keys())}")
    name = _resolve_name(pipeline, name, fn)
    items = list(pipeline.items())
    items.insert(list(pipeline.keys()).index(anchor) + offset, (name, fn))
    pipeline.clear()
    pipeline.update(OrderedDict(items))


def _resolve_name(pipeline: OrderedDict[str, F], name: str | None, fn: F) -> str:
    assert callable(fn), f"Expected fn to be callable, got {fn}"
    if name is None:
        name = getattr(fn, "__name__", None)
    if name is None:  # E.g. functools.partial objects have no __name__
        raise ValueError(f"{fn} has no __name__, an explicit name is required")
    if name in pipeline:
        raise KeyError(f"Pipeline stage '{name}' already exists. Names must be unique")
    assert isinstance(name, str), f"Expected name to be str, got {name}"
    return name
