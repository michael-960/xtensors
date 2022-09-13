from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING, Callable, Literal, Sequence, Tuple, overload

from ..typing import Dims, Coords
from ..basic_utils import permute, align, shapes_broadcastable, mergecoords, mergedims

from ._types import Broadcaster, Dimcaster, DimMerger, CoordMerger

from ._dimcast import unilateral_dimcast, trivial_dimcast

from ._template import Template


if TYPE_CHECKING:
    from .._base import XTensor



class TensorBroadcastError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def broadcaster_wrapper(
        dimcast: Dimcaster,
        dimmerge: DimMerger, coordmerge: CoordMerger,
        shapecheck: bool=True) -> Broadcaster:

    def _broadcast(X: XTensor, Y: XTensor):
        return broadcast(X, Y, dimcast, dimmerge, coordmerge, shapecheck)
    return _broadcast


def broadcast(X: XTensor, Y: XTensor, 
        dimcast: Dimcaster, dimmerge: DimMerger, coordmerge: CoordMerger,
        shapecheck: bool=True
        ) -> Tuple[np.ndarray, np.ndarray, Dims, Coords]:
    '''
    Given two named tensors, return two numpy ND arrays of the same rank that
    can be broadcast together, along with the dimension names and coordinates
    after broadcasting.
    '''


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


unilateral_broadcaster = broadcaster_wrapper(
            dimcast=unilateral_dimcast(strict=False),
            dimmerge=mergedims, coordmerge=mergecoords
)
'''
This broadcaster takes the second tensor and permutes its dimensions to match
the first one. The algorithm for finding the permutation is defined by
unilateral_dimcast()
'''


vanilla_broadcaster = broadcaster_wrapper(
            dimcast=trivial_dimcast,
            dimmerge=mergedims, coordmerge=mergecoords
)
'''
Vanilla broadcaster: the same as what's used in torch's named tensors.
Dimension names (or None) have to match at the same axis position.
'''


def template_broadcaster(dims: Sequence[str|int], channels: Sequence[Literal['x','y']|None]) -> Broadcaster:
    '''
    Return a template broadcaster with the specified dimensions and channels
    '''

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
def cast(broadcaster: Broadcaster, X: XTensor) -> Callable[[XTensor], XTensor]: ...
@overload
def cast(broadcaster: Broadcaster, X: XTensor, Y: XTensor) -> XTensor: ...

def cast(broadcaster: Broadcaster, X: XTensor, Y: XTensor|None=None):
    if Y is None:
        def _cast(Y1: XTensor, /):
            x, _y, dims, coords = broadcaster(X, Y1)

            coords = [(coord if _y.shape[axis] > 1 else None) 
                    for axis, coord in enumerate(coords)]

            return Y1.__class__(_y, dims, coords)
        return _cast
    return cast(broadcaster, X)(Y)

