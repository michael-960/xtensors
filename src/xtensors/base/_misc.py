from __future__ import annotations

from typing import Any, List, Protocol, Tuple, cast, Union
import numpy as np

import xarray as xr
from xarray import DataArray
from xtensors.typing import NDArray, number, DimLike

from ._broadcast import broadcast_arrays

from xtensors.unify import get_axis_from_dim, strip_dims, get_axes

from scipy import special


def where(condition: NDArray, x: NDArray|number, y: NDArray|number) -> DataArray:
    '''
        The returned tensor is named only if condition is named
    '''
    dims = None

    X = cast(Union[NDArray,np.number], x)
    Y = cast(Union[NDArray,np.number], y)

    _c = condition.__array__()
    _x = X
    _y = Y

    if hasattr(x, 'shape'):
        _x = cast(NDArray, X.__array__())
        _c, _x, _, _ = broadcast_arrays(_c, _x)

    if hasattr(y, 'shape'):
        _y = cast(NDArray, Y.__array__())
        _c, _y, _, _ = broadcast_arrays(_c, _y)

    _z = np.where(_c, _x, _y)

    if isinstance(condition, DataArray): 
        dims = list(condition.dims)
        if len(_z.shape) > len(dims):
            dims = ['dim_i' for i in range(len(_z.shape)-len(dims))] + dims

    return DataArray(_z, dims=dims)


def softmax(x: NDArray, dim: DimLike) -> DataArray:

    axis = get_axes(x, dim)[0]
    _y = special.softmax(x.__array__(), axis=axis)

    dims = None
    if isinstance(x, DataArray):
        dims = list(x.dims)

    return DataArray(_y, dims=dims)



def get_rank(x: NDArray|list|number) -> int:
    if hasattr(x, 'shape'):
        return len(cast(NDArray, x).shape)

    if isinstance(x, list):
        return get_rank(x[0]) + 1

    return 0


