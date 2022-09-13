from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Sequence
import numpy as np

from ..typing import Coords
if TYPE_CHECKING: from .._base import XTensor


def mergecoords(X: XTensor|Coords, Y: XTensor|Coords, rtol: float=1e-8, atol: float=1e-8) -> Coords:
    '''
    '''

    from .._base import XTensor

    if isinstance(X, XTensor): coords_x = list(X.coords)
    else: coords_x = X

    if isinstance(Y, XTensor): coords_y = list(Y.coords)
    else: coords_y = Y

    rank_x, rank_y = len(coords_x), len(coords_y)

    if rank_x < rank_y: return mergecoords(Y, X)


    newcoords: Coords = []

    coords_y: Coords = [None for _ in range(rank_y, rank_x)] + coords_y
    
    for coord_x, coord_y in zip(coords_x, coords_y):
        if coord_x is not None and coord_y is not None:

            try: 
                condition = np.allclose(coord_x, coord_y, rtol=rtol, atol=atol)
            except TypeError:
                condition = (coord_x == coord_y)

            if not condition:
                raise ValueError(f'Incompatible coordinates')

        newcoords.append(list(coord_x) if coord_x is not None else list(coord_y) if coord_y is not None else None)

    return newcoords

