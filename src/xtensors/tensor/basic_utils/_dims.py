from __future__ import annotations
from typing import List, Literal, Sequence
from .._base import XTensor
from ..typing import Dims

from ... import numpy as xtnp

from ._axes import permute



def mergedims(X: XTensor, Y: XTensor) -> Dims:
    '''
    '''

    if X.rank < Y.rank: return mergedims(Y, X)

    newdims = []

    dims_x = list(X.dims)
    dims_y = [None for _ in range(Y.rank, X.rank)] + list(Y.dims)
    
    for dim_x, dim_y in zip(dims_x, dims_y):
        if dim_x is not None and dim_y is not None and dim_x != dim_y:
            raise ValueError(f'Incompatible dimensions: {X.dims} and {Y.dims}')
        newdims.append(dim_x if dim_x is not None else dim_y)

    return newdims


def dimslast(X: XTensor, dims: Sequence[str]) -> XTensor:
    '''

    '''
    axes: List[int] = [X.get_axis(dim) for dim in dims]
    other_axes: List[int|None] = [axis for axis in range(X.rank) if axis not in axes]

    return permute(X, other_axes+axes)


def dimsfirst(X: XTensor, dims: Sequence[str]) -> XTensor:
    '''

    '''
    axes: List[int] = [X.get_axis(dim) for dim in dims]
    other_axes: List[int|None] = [axis for axis in range(X.rank) if axis not in axes]

    return permute(X, axes+other_axes)


def flatten(X: XTensor, dims: Sequence[str], dim_out: str|None, position: Literal['left', 'right']='right'):
    x = X.data
    axes = [X.get_axis(dim) for dim in dims]

    remaining_dims = [dim for axis, dim in enumerate(X.dims ) if axis not in axes]
    remaining_coords = [coord for axis, coord in enumerate(X.coords) if axis not in axes]


    # TODO: implement coordinate meshgrid
    coord_out = None

    x_flat = xtnp.flatten(x, axes, position=position)

    if position == 'left':
        Y = XTensor(x_flat, [dim_out]+remaining_dims, [coord_out] + remaining_coords)
        return Y

    else:
        Y = XTensor(x_flat, remaining_dims+[dim_out], remaining_coords+[coord_out])
        return Y



