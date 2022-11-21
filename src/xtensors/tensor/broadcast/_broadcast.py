from __future__ import annotations
from functools import wraps
import numpy as np


from typing import TYPE_CHECKING, overload

from ..basic_utils import permute, align, shapes_broadcastable, mergecoords, mergedims, copy_sig


from ._dimcast import unilateral_dimcast, trivial_dimcast

from ._template import Template


if TYPE_CHECKING:
    from typing import Callable, Literal, Sequence, Tuple
    from ..typing import Dims, Coords
    from .._base import XTensor

from ._types import Broadcaster, Dimcaster, DimMerger, CoordMerger



class TensorBroadcastError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def broadcaster_wrapper(
        dimcast: Dimcaster,
        dimmerge: DimMerger, coordmerge: CoordMerger,
        shapecheck: bool=True) -> Broadcaster:

    def _broadcast(X: XTensor, Y: XTensor) -> Tuple[np.ndarray, np.ndarray, Dims, Coords]:
        return broadcast(X, Y, dimcast, dimmerge, coordmerge, shapecheck)
    return _broadcast


def broadcast(
    X: XTensor, Y: XTensor, 
    dimcast: Dimcaster, dimmerge: DimMerger, coordmerge: CoordMerger,
    shapecheck: bool=True
) -> Tuple[np.ndarray, np.ndarray, Dims, Coords]:
    r"""

    Given two named tensors, return two numpy ND arrays of the same rank that
    can be broadcast together, along with the dimension names and coordinates
    after broadcasting.

    :param X,Y: :py:class:`xtensors.XTensor` objects to be broadcast together
    :param dimcast: An object implementing the :py:class:`xtensors.Dimcaster` protocol
    :param dimmerge: An object implmenting the :py:class:`xtensors.DimMerger` protocol
    :param coordmerge: An object implementing the :py:class:`xtensors.CoordMerger` protocol

    """

    axes_x, axes_y = dimcast(X, Y)

    X1 = permute(X, axes_x)
    Y1 = permute(Y, axes_y)

    X1, Y1 = align(X1, Y1)

    newdims = dimmerge(X1, Y1)
    newcoords = coordmerge(X1, Y1)

    if shapecheck:
        if not shapes_broadcastable(X1.shape, Y1.shape):
            raise TensorBroadcastError(
                    f'Broadcast impossible with shapes and dims <{X.dims},{X.shape}> and <{Y.dims},{Y.shape}>')

    return X1.data, Y1.data, newdims, newcoords


_unilateral_broadcaster = broadcaster_wrapper(
        dimcast=unilateral_dimcast(strict=False),
        dimmerge=mergedims,
        coordmerge=mergecoords
)

@copy_sig(_unilateral_broadcaster)
def unilateral_broadcaster(X: XTensor, Y: XTensor):
    """
    This broadcaster takes the second tensor and permutes its dimensions to match
    the first one. The algorithm for finding the permutation is defined by
    unilateral_dimcast()
    """
    return _unilateral_broadcaster(X, Y)


_vanilla_broadcaster = broadcaster_wrapper(
        dimcast=trivial_dimcast,
        dimmerge=mergedims,
        coordmerge=mergecoords
)

@copy_sig(_vanilla_broadcaster)
def vanilla_broadcaster(X: XTensor, Y: XTensor):
    """
    Vanilla broadcaster: the same as what's used in torch's named tensors.
    Dimension names (or None) have to match at the same axis position.
    """
    return _vanilla_broadcaster(X, Y)


def template_broadcaster(dims: Sequence[str|int], channels: Sequence[Literal['x','y']|None]) -> Broadcaster:
    """
    Return a template broadcaster with the specified dimensions and channels
    """

    template = Template.from_dims_channels(dims, channels)
    
    def _broadcast(X: XTensor, Y: XTensor):
        X1 = template.cast_and_update(X, 'x')
        Y1 = template.cast_and_update(Y, 'y')

        dims, coords = template.dims, template.coords
        template.clear()
        return X1.data, Y1.data, dims, coords

    # _broadcast.cast = lambda x, y: y

    return _broadcast


@overload
def cast(broadcaster: Broadcaster, X: XTensor) -> Callable[[XTensor], XTensor]:
    ...

@overload
def cast(broadcaster: Broadcaster, X: XTensor, Y: XTensor) -> XTensor:
    ...

def cast(broadcaster: Broadcaster, X: XTensor, Y: XTensor|None=None):
    """
    Broadcast the second tensor with :code:`broadcaster(X, Y)`. If :code:`Y` is
    :code:`None` or not provided, a casting function is returned instead.
    """
    if Y is None:
        def _cast(Y1: XTensor, /):
            x, _y, dims, coords = broadcaster(X, Y1)

            coords = [(coord if _y.shape[axis] > 1 else None) 
                    for axis, coord in enumerate(coords)]

            return Y1.__class__(_y, dims, coords)
        return _cast
    return cast(broadcaster, X)(Y)




