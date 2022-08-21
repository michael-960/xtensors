from __future__ import annotations
from typing import List, Tuple, cast
import numpy as np
from xarray import DataArray

from xtensors.typing import AxisDim, AxisDimPair
from xtensors.typing import NDArray


def dims_to_axes(dims: Tuple[str,...], target_dims: Tuple[str,...]) -> Tuple[int,...]:
    res: List[int] = []
    
    _dims = np.array(dims)

    for tdim in target_dims:
        index = np.where(_dims == tdim)[0].item()
        res.append(index)
    return tuple(res)


def get_axis(x: NDArray,
        dim: AxisDim|Tuple[str,...]|Tuple[int,...]|AxisDimPair|Tuple[AxisDimPair,...]|None) -> Tuple[int, ...]:

    if isinstance(x, DataArray):
        _dims = cast(Tuple[str,...], x.dims)
        if dim is None:
            return tuple([i for i in range(len(_dims))])

        if isinstance(dim, int):
            return (dim,)

        if isinstance(dim, str):
            return dims_to_axes(_dims, (dim,))

        if isinstance(dim, tuple):
            if isinstance(dim[0], int):
                return cast(Tuple[int,...], dim)

            if isinstance(dim[0], str):
                return dims_to_axes(_dims, cast(Tuple[str,...], dim))

            if isinstance(dim[0], tuple):
                dim = cast(Tuple[AxisDimPair,...], dim)
                return dims_to_axes(_dims, tuple([d[1] for d in dim]))

    if isinstance(dim, int):
        return (dim,)

    if isinstance(dim, tuple):
        if isinstance(dim[0], int):
            return cast(Tuple[int,...], dim)

        if isinstance(dim[0], tuple):
            dim = cast(Tuple[AxisDimPair,...], dim)
            return tuple([d[0] for d in dim])

    raise ValueError


def strip_dims(olddims: Tuple[str, ...], axis: Tuple[int, ...]) -> Tuple[str,...]:
    newdims = list(olddims)
    stripped_dims = [olddims[i] for i in axis] 

    for sd in stripped_dims: newdims.remove(sd)

    return tuple(newdims)
     





