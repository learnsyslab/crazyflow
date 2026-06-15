"""This file is to be remove later as soon as a proper typing is available by the official array-api."""

from typing import Any, TypeAlias

import numpy.typing as npt

Array: TypeAlias = Any  # To be changed to array_api_typing later
ArrayLike: TypeAlias = Array | npt.ArrayLike
