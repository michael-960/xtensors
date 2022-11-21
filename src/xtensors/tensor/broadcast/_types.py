from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Protocol, Sequence, Tuple
import numpy as np


if TYPE_CHECKING:
    from ...tensor._base import XTensor
    from ..typing import AxesPermutation, Coords, Dims


class Dimcaster(Protocol):
    """
    The dimension caster protocol.
    """
    def __call__(self, X: XTensor, Y: XTensor) -> Tuple[AxesPermutation, AxesPermutation]:
        r"""
        :param x,y: A pair of :py:class:`xtensors.XTensor` objects 

        :return: A pair of :py:class:`xtensors.AxesPermutation` (i.e. lists of ints) 
                 specifying how the axes of :code:`X` and :code:`Y` should be permuted.
                 
        """
        ...


class DimMerger(Protocol):
    """
    Dimension-name-merging protocol
    """
    def __call__(self, X: XTensor, Y: XTensor) -> Dims:
        """
        Merge the dimension names of :code:`X` and :code:`Y`

        :rtype: :py:class:`xtensors.Dims`
         
        """
        ...


class CoordMerger(Protocol):
    """
    Coordinates-mergin protocol
    """
    def __call__(self, X: XTensor, Y: XTensor) -> Coords:
        """
        Merge the coordinates of of :code:`X` and :code:`Y`

        :rtype: :py:class:`xtensors.Coords`

        """
        ...


class Broadcaster(Protocol):
    """
    Broadcaster protocol
    """
    def __call__(
        self,
        X: XTensor, Y: XTensor
    ) -> Tuple[np.ndarray, np.ndarray, Dims, Coords]:
        """
        Broadcast two tensors together

        :return: The resulting :code:`np.ndarray` s that are mutually
                broadcastable, and the new dimension names as well as the coordinates
                after broadcasting.

        """
        ...






