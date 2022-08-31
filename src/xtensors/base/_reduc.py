from __future__ import annotations
from typing import Any, Protocol, Sequence, Tuple, overload, cast
import numpy as np
from xarray import DataArray

from xtensors.typing import NDArray
from xtensors.typing import AxisDimPair, DimLike, DimsLike

from xtensors.unify import get_axes, strip_dims

'''
For reduction functions that can act over multiple axes/dims.
'''


class _np_reduction_func(Protocol):
    def __call__(self, a: NDArray, axis: int|Tuple[int,...]) -> DataArray: ...


class ReductionFunc(Protocol):
    def __call__(self, x: NDArray, dim: DimLike|DimsLike|None) -> DataArray: ...




def _reduction_factory(_np_func: _np_reduction_func) -> ReductionFunc:
    def _reduce(x: NDArray, dim: DimLike|DimsLike|None=None) -> DataArray:

        _x = x.__array__()
        _axis = get_axes(x, dim)

        newdims = None
        coords_map = dict()
        if isinstance(x, DataArray):
            olddims = cast(Tuple[str,...], x.dims)
            newdims = strip_dims(olddims, _axis)
            for dimkey in newdims:
                coords_map[dimkey] = x.coords[dimkey]

        _y = _np_func(_x, axis=_axis)

        return DataArray(_y, dims=newdims, coords=coords_map)
    return _reduce


_sum = _reduction_factory(np.sum)
_mean = _reduction_factory(np.mean)
_std = _reduction_factory(np.std)

_nanmean = _reduction_factory(np.nanmean)
_nanstd = _reduction_factory(np.nanstd)
_nansum = _reduction_factory(np.nansum)

_max = _reduction_factory(np.max)
_min = _reduction_factory(np.min)

_nanmax = _reduction_factory(np.nanmax)
_nanmin = _reduction_factory(np.nanmin)

_all = _reduction_factory(np.all)
_any = _reduction_factory(np.any)



