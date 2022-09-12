from __future__ import annotations
import numpy as np

from typing import Tuple

from .._base import XTensor
from ..typing import Dims, Coords
from ..basic_utils import permute, align, shapes_broadcastable

from ._types import Broadcaster, Dimcaster, DimMerger, CoordMerger



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





