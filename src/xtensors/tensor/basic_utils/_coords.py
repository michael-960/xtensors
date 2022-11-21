from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .._base import XTensor
    from ..typing import Coords


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
    
    if not coords_same(coords_x, coords_y, rtol=rtol, atol=atol, none_compatible=True):
        raise ValueError('Coordinates incompatible')

    for coord_x, coord_y in zip(coords_x, coords_y):
        newcoords.append(coord_x if coord_x is not None else coord_y if coord_y is not None else None)

    return newcoords


def coords_same(coords1: Coords, coords2: Coords, /, *,
        rtol: float=1e-8, atol: float=1e-8, none_compatible: bool=False) -> bool:
    """
    Return whether the given coordinates are the same.
    Two coordinate sequences are considered the same if:
        - 
    """
    
    if len(coords1) != len(coords2): return False

    for coord1, coord2 in zip(coords1, coords2):
        if coord1 is None or coord2 is None: 
            condition = (coord1 is None and coord2 is None) or none_compatible
        else:
            try: 
                condition = np.allclose(coord1, coord2, rtol=rtol, atol=atol)
            except TypeError:
                condition = (coord1 == coord2)

        if not condition: return False

    return True
