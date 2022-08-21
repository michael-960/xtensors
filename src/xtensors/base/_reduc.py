from __future__ import annotations
from typing import Any, Protocol, Sequence, Tuple, overload, cast
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from torch.types import Number
from xarray import DataArray

from xtensors.typing import NDArray
from xtensors.typing import AxisDim, AxisDimPair

from xtensors.unify import get_axis, strip_dims


class _np_reduction_func(Protocol):
    def __call__(self, a: NDArray, axis: int|Tuple[int,...]=...) -> DataArray: ...


class ReductionFunc(Protocol):
    @overload
    def __call__(self, x: DataArray, dim: str|Tuple[str,...]|None) -> DataArray: ...
    @overload
    def __call__(self, x: DataArray, dim: int|Tuple[int,...]|None) -> DataArray: ...
    @overload
    def __call__(self, x: NDArray, dim: int|Tuple[int,...]|Tuple[int,str]|Tuple[Tuple[int,str],...]|None) -> DataArray: ...

    def __call__(self, x: NDArray, dim: str|int|Tuple[str,...]|Tuple[int,...]|Tuple[int,str]|Tuple[Tuple[int,str],...]|None) -> DataArray: ...


def _reduction_factory(_np_func: _np_reduction_func) -> ReductionFunc:
    def _reduce(x: NDArray, dim: AxisDim | Tuple[str,...] | Tuple[int,...] | AxisDimPair | Tuple[AxisDimPair, ...] | None=None) -> DataArray:

        _x = x.__array__()
        _axis = get_axis(x, dim)

        newdims = None
        if isinstance(x, DataArray):
            olddims = cast(Tuple[str,...], x.dims)
            newdims = strip_dims(olddims, _axis)

        _y = _np_func(_x, axis=_axis)

        return DataArray(_y, dims=newdims)

        # if dim is None:
        #     return _np_func(_x)
        #
        # if not isinstance(dim, tuple): 
        #     # dim is str or int
        #     _dims = (dim,)
        # else:
        #     # dim is tuple
        #     if isinstance(dim[0], tuple):
        #         # dim is ((int,str), (int,str),...)
        #         dim = cast(Tuple[Tuple[int,str],...], dim)
        #         if isinstance(x, DataArray):
        #             _dims = tuple([_dim[1] for _dim in dim])
        #         else:
        #             _dims = tuple([_dim[0] for _dim in dim])
        #
        #     elif isinstance(dim[0], int) and isinstance(dim[1], str):
        #         # dim is (int,str)
        #         _dims = (dim[1],) if isinstance(x, DataArray) else (dim[0],)
        #     else:
        #         # dim is (int, int, ...) or (str, str, ...)
        #         _dims = dim
        #
        # if isinstance(x, DataArray):
        #     olddims = np.array(x.dims)
        #     newdims = list(x.dims)
        #     _x = x.data
        #
        #     if isinstance(_dims[0], str):
        #         for _dim in _dims: newdims.remove(_dim)
        #         _axes = tuple([np.where(olddims == _dim)[0].item() for _dim in _dims])
        #        
        #     else:
        #         _dims = cast(Tuple[int], _dims)
        #         for _dim in _dims: newdims.remove(olddims[_dim])
        #         _axes = _dims
        # else:
        #     newdims = None
        #     _dims = cast(Tuple[int], _dims)
        #     _axes = _dims
        #    
        #     if not isinstance(x, np.ndarray):
        #         _x = x.__array__()
        #     else:
        #         _x = x
        #
        # y = _np_func(_x, axis=_axes) 
        # return DataArray(y, dims=newdims)
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

