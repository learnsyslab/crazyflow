"""Named, ordered pipelines of simulation functions.

At its core, the pipeline is just an OrderedDict of functions with some helper methods for
inserting functions before and after existing elements by name, i.e. keys.

``Pipeline`` stores its functions under unique names so that stages can be addressed directly,
e.g. ``pipeline.insert_before("integration", my_fn)``, instead of through positional indices.
Stage names default to the function's ``__name__`` and can be set explicitly for callables that
don't have one (e.g. ``functools.partial`` objects) by passing a ``(name, fn)`` tuple.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Generator, Iterable, Iterator

F = TypeVar("F", bound=Callable)


class Pipeline(Generic[F], Mapping[str, F]):
    """Ordered collection of named functions.

    Functions are addressed by their unique name, which allows inserting, replacing and removing
    stages without relying on their position in the pipeline. Iterating over a pipeline yields the
    functions in order, ready to be chained at trace time.

    Pipelines support summation like tuples: ``pipeline + fn`` and ``fn + pipeline`` return
    new pipelines with the functions appended or prepended.

    Warning:
        Modifying a pipeline does not affect the compiled simulation functions. Always rebuild
        with ``Sim.build_step_fn()`` / ``Sim.build_reset_fn()`` after changing a pipeline.
    """

    def __init__(self, items: Iterable[tuple[str, F]] = ()):
        self._entries: OrderedDict[str, F] = {}
        self.extend(items)

    @property
    def names(self) -> tuple[str, ...]:
        """The names of all stages in pipeline order."""
        return tuple(self.keys())

    def append(self, fn: F, name: str | None = None):
        """Add a function to the end of the pipeline.

        Args:
            fn: The function to add.
            name: Unique name of the new stage. Defaults to ``fn.__name__``.
        """
        name, fn = self._make_entry(name, fn)
        self._entries[name] = fn

    def prepend(self, fn: F, name: str | None = None):
        """Add a function to the front of the pipeline.

        Args:
            fn: The function to add.
            name: Unique name of the new stage. Defaults to ``fn.__name__``.
        """
        name, fn = self._make_entry(name, fn)
        self._entries = OrderedDict([(name, fn)]) | self._entries

    def extend(self, items: Iterable[tuple[str, F] | F] | Pipeline[F]) -> None:
        """Add multiple functions to the end of the pipeline.

        Args:
            items: An iterable of functions or ``(name, fn)`` tuples.
        """
        if isinstance(items, Pipeline):
            return self.extend(items.items())
        for item in items:
            name, fn = item if isinstance(item, tuple) else (None, item)
            self.append(fn, name)

    def insert_before(self, anchor: str, fn: F, name: str | None = None):
        """Insert a function directly before the stage named ``anchor``.

        Args:
            anchor: Name of an existing stage to insert before.
            fn: The function to insert.
            name: Unique name of the new stage. Defaults to ``fn.__name__``.
        """
        name, fn = self._make_entry(name, fn)
        index = self.index(anchor)
        items = list(self.items())
        items.insert(index, (name, fn))
        self._entries = OrderedDict(items)

    def insert_after(self, anchor: str, fn: F, name: str | None = None):
        """Insert a function directly after the stage named ``anchor``.

        Args:
            anchor: Name of an existing stage to insert after.
            fn: The function to insert.
            name: Unique name of the new stage. Defaults to ``fn.__name__``.
        """
        name, fn = self._make_entry(name, fn)
        index = self.index(anchor)
        items = list(self.items())
        items.insert(index + 1, (name, fn))
        self._entries = OrderedDict(items)

    def replace(self, name: str, fn: F):
        """Replace the function of the stage named ``name``, keeping its position and name.

        Args:
            name: Name of the stage to replace.
            fn: The new function for the stage.
        """
        self._entries[name] = fn

    def remove(self, name: str):
        """Remove the stage named ``name`` from the pipeline.

        Args:
            name: Name of the stage to remove.
        """
        del self._entries[name]

    def index(self, name: str) -> int:
        """Return the position of the stage named ``name``.

        Args:
            name: Name of the stage to look up.

        Raises:
            ValueError: If no stage with this name exists.
        """
        for i, key in enumerate(self._entries):
            if key == name:
                return i
        raise ValueError(f"No pipeline stage named '{name}'. Available stages: {self.names}")

    def _make_entry(self, name: str | None, fn: F) -> tuple[str, F]:
        if name is None:
            name = getattr(fn, "__name__", None)
        if name is None:  # E.g. functools.partial objects have no __name__
            raise ValueError(f"{fn} has no __name__, an explicit name is required")
        if name in self.keys():
            raise KeyError(f"Pipeline stage '{name}' already exists. Names must be unique")
        assert isinstance(name, str), f"Expected name to be str, got {name}"
        assert callable(fn), f"Expected fn to be callable, got {fn}"
        return (name, fn)

    def __add__(self, other: F | tuple[str, F] | Pipeline[F]) -> Pipeline[F]:
        """Return a new pipeline with the items of ``other`` appended."""
        new = Pipeline(self)
        if isinstance(other, Pipeline):
            new.extend(other.items())
            return new
        if not isinstance(other, tuple):
            other = self._make_entry(None, other)
        assert len(other) == 2, f"Invalid pipeline sum operand: {other}"
        assert isinstance(other[0], str), f"Expected name to be str, got {other[0]}"
        assert callable(other[1]), f"Expected fn to be callable, got {other[1]}"
        new.append(other[1], other[0])
        return new

    def __radd__(self, other: Iterable[F]) -> Pipeline[F]:
        """Return a new pipeline with the items of ``other`` prepended."""
        new = Pipeline(other)
        new.extend(self)
        return new

    def __iter__(self) -> Iterator[F]:
        """Iterate over the pipeline functions in order."""
        return (fn for fn in self.values())

    def __len__(self) -> int:
        """Return the number of stages in the pipeline."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check whether a stage with the given name exists in the pipeline."""
        return name in self.keys()

    def __repr__(self) -> str:
        """Show the stage names in pipeline order."""
        return f"{type(self).__name__}({' -> '.join(self.names)})"

    def __getitem__(self, name: str) -> F:
        """Get the function of the stage named ``name``.

        Raises:
            KeyError: If no stage with this name exists.
        """
        return self._entries[name]

    def keys(self) -> tuple[str, ...]:
        """Return the stage names in pipeline order."""
        return tuple(self._entries.keys())

    def values(self) -> tuple[F, ...]:
        """Return the stage functions in pipeline order."""
        return tuple(self._entries.values())

    def items(self) -> Generator[tuple[str, F], None, None]:
        """Return the stage names and functions in pipeline order."""
        return self._entries.items()
