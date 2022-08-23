from __future__ import annotations
from typing import Any, Protocol, Sequence, Tuple, overload, cast
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from torch.types import Number
from xarray import DataArray

from xtensors.typing import NDArray
from xtensors.typing import AxisDimPair, DimLike, DimsLike

from xtensors.unify import get_axes, strip_dims


class _np_reduction_func(Protocol):
    def __call__(self, a: NDArray, axis: int|Tuple[int,...]) -> DataArray: ...


class ReductionFunc(Protocol):
    def __call__(self, x: NDArray, dim: DimLike|DimsLike|None) -> DataArray: ...


def _reduction_factory(_np_func: _np_reduction_func) -> ReductionFunc:
    def _reduce(x: NDArray, dim: DimLike|DimsLike|None=None) -> DataArray:

        _x = x.__array__()
        _axis = get_axes(x, dim)

        newdims = None
        if isinstance(x, DataArray):
            olddims = cast(Tuple[str,...], x.dims)
            newdims = strip_dims(olddims, _axis)

        _y = _np_func(_x, axis=_axis)

        return DataArray(_y, dims=newdims)
    return _reduce


_sum = _reduction_factory(np.sum)

_mean = _reduction_factory(np.mean)
_std = _reduction_factory(np.std)

_max = _reduction_factory(np.max)
_min = _reduction_factory(np.min)

_all = _reduction_factory(np.all)
_any = _reduction_factory(np.any)

_nanmean = _reduction_factory(np.nanmean)
_nanstd = _reduction_factory(np.nanstd)
_nansum = _reduction_factory(np.nansum)



