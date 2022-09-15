from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Protocol, Sequence, Tuple
import numpy as np


if TYPE_CHECKING:
    from ...tensor._base import XTensor

from ..typing import AxesPermutation, Coords, Dims


class Dimcaster(Protocol):
    def __call__(self, X: XTensor, Y: XTensor) -> Tuple[AxesPermutation, AxesPermutation]: ...


class DimMerger(Protocol):
    def __call__(self, X: XTensor, Y: XTensor) -> Dims: ...


class CoordMerger(Protocol):
    def __call__(self, X: XTensor, Y: XTensor) -> Coords: ...


class Broadcaster(Protocol):
    def __call__(self,
            X: XTensor, Y: XTensor) -> Tuple[np.ndarray, np.ndarray, Dims, Coords]: ...

