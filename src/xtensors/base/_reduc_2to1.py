from __future__ import annotations
from typing import cast, overload, Tuple, Protocol, Any

import numpy as np

from xarray import DataArray

from xtensors.typing import NDArray
from xtensors.typing import DimLike, DimsLike

from xtensors.unify import get_axes, strip_dims



def diagonal(x: NDArray, dim1: DimLike, dim2: DimLike, dim_out: str) -> DataArray:

    assert isinstance(dim1, (str, int, tuple))
    _x = x.__array__()

    axes1 =  get_axes(x, dim=dim1)
    axes2 =  get_axes(x, dim=dim2)

    assert len(axes1) == len(axes2) == 1

    axis1 = axes1[0]
    axis2 = axes2[0]

    y = np.diagonal(_x, axis1=axis1, axis2=axis2)

    if isinstance(x, DataArray):
        olddims = cast(Tuple[str], x.dims)
        newdims = list(strip_dims(olddims, (axis1, axis2)))
        newdims.append(dim_out)
        return DataArray(y, dims=newdims)
    else:
        y_ = DataArray(y)
        y_ = y_.rename({y_.dims[-1]: dim_out})
        return y_



