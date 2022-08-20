from __future__ import annotations
from typing import cast, overload, Tuple, Protocol, Any

import numpy as np
from numpy.typing import ArrayLike

from xarray import DataArray


@overload
def diagonal(x: DataArray, dim1: str, dim2: str, dim_out: str) -> DataArray: ...
@overload
def diagonal(x: DataArray, dim1: int, dim2: int, dim_out: str) -> DataArray: ...
@overload
def diagonal(x: ArrayLike, dim1: int, dim2: int, dim_out: str) -> DataArray: ...
@overload
def diagonal(x: ArrayLike, dim1: Tuple[int,str], dim2: Tuple[int,str], dim_out: str) -> DataArray: ...

def diagonal(
        x: ArrayLike,
        dim1: str|int|Tuple[int,str], dim2: str|int|Tuple[int,str],
        dim_out: str) -> DataArray:

    assert isinstance(dim1, (str, int, tuple))
    assert type(dim1) is type(dim2)


    if isinstance(x, DataArray):
        _x = x.data
    else:
        _x = x


    if isinstance(dim1, int) and isinstance(dim2, int):
        axis1 = dim1
        axis2 = dim2

    elif isinstance(x, DataArray):
        if isinstance(dim2, tuple) and isinstance(dim1, tuple):
            _dim1 = dim1[1]
            _dim2 = dim2[1]
        else: # dim is string
            _dim1 = dim1
            _dim2 = dim2

        dims = np.array(x.dims)
        axis1 = np.where(dims == _dim1)[0].item()
        axis2 = np.where(dims == _dim2)[0].item()

    elif isinstance(dim2, tuple) and isinstance(dim1, tuple):
        # not xarray, not int -> should be tuple[int, str]
        axis1 = dim1[0]
        axis2 = dim2[0]

    else:
        raise ValueError('Invalid combination of input types')

    y = np.diagonal(_x, axis1=axis1, axis2=axis2)

    if isinstance(x, DataArray):
        newdims = list(x.dims)
        if isinstance(dim1, str) and isinstance(dim2, str):
            newdims.remove(dim1)
            newdims.remove(dim2)
        elif isinstance(dim1, int) and isinstance(dim2, int):
            newdims.remove(x.dims[dim1])
            newdims.remove(x.dims[dim2])
        else:
            dim1 = cast(Tuple[int, str], dim1)
            dim2 = cast(Tuple[int, str], dim2)
            newdims.remove(dim1[1])
            newdims.remove(dim2[1])
            
        newdims.append(dim_out)
        return DataArray(y, dims=newdims)
    else:
        y_ = DataArray(y)
        y_ = y_.rename({y_.dims[-1]: dim_out})

        return y_



