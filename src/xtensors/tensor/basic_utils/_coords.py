from typing import Any, List, Sequence
import numpy as np
from .._base import XTensor

from ..typing import Coords



def mergecoords(X: XTensor, Y: XTensor, rtol: float=1e-8, atol: float=1e-8) -> Coords:
    '''
    '''

    if X.rank < Y.rank: return mergecoords(Y, X)

    newcoords: Coords = []

    coords_x: Coords = list(X.coords)
    coords_y: Coords = [None for _ in range(Y.rank, X.rank)] + list(Y.coords)
    
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



