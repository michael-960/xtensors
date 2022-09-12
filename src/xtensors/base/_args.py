from __future__ import annotations

from xarray import DataArray
from ..typing import NDArray, DimLike, DimsLike

from ..unify import get_axes, strip_dims, to_dataarray

from .. import numpy as xtnp



def argsmin(x: NDArray, dim: DimLike|DimsLike|None=None):

    _x = x.__array__()
    x_ = to_dataarray(x)



    axes = get_axes(x, dim)

    dims = strip_dims(x_.dims, axes)

    _args = xtnp.argsmin(_x, axes=axes)

    return DataArray(_args, )
