from __future__ import annotations
from typing import Any, Protocol, Sequence, Tuple, overload, cast
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from xarray import DataArray


class _np_reduction_func(Protocol):
    def __call__(self, a: ArrayLike, axis: int|Tuple[int,...]=...) -> Any: ...

class ReductionFunc(Protocol):
    @overload
    def __call__(self, x: DataArray, dim: str|Tuple[str,...]|None) -> DataArray: ...
    @overload
    def __call__(self, x: DataArray, dim: int|Tuple[int,...]|None) -> DataArray: ...
    @overload
    def __call__(self, x: ArrayLike, dim: int|Tuple[int,...]|Tuple[int,str]|Tuple[Tuple[int,str],...]|None) -> DataArray: ...

    def __call__(self, x: ArrayLike, dim: str|int|Tuple[str,...]|Tuple[int,...]|Tuple[int,str]|Tuple[Tuple[int,str],...]|None) -> DataArray: ...




def _reduction_factory(_np_func: _np_reduction_func) -> ReductionFunc:
    def _reduce(x: ArrayLike, dim: str|int | Tuple[str,...] | Tuple[int,...] | Tuple[int,str] | Tuple[Tuple[int,str],...] | None=None) -> DataArray:
        if dim is None:
            if isinstance(x, DataArray):
                _x = x.data
            else:
                _x = x
            return _np_func(_x)
        
        if not isinstance(dim, tuple): 
            # dim is str or int
            _dims = (dim,)
        else:
            # dim is tuple
            if isinstance(dim[0], tuple):
                # dim is ((int,str), (int,str),...)
                dim = cast(Tuple[Tuple[int,str],...], dim)
                if isinstance(x, DataArray):
                    _dims = tuple([_dim[1] for _dim in dim])
                else:
                    _dims = tuple([_dim[0] for _dim in dim])

            elif isinstance(dim[0], int) and isinstance(dim[1], str):
                # dim is (int,str)
                _dims = (dim[1],) if isinstance(x, DataArray) else (dim[0],)
            else:
                # dim is (int, int, ...) or (str, str, ...)
                _dims = dim

        if isinstance(x, DataArray):
            olddims = np.array(x.dims)
            newdims = list(x.dims)
            _x = x.data

            if isinstance(_dims[0], str):
                for _dim in _dims: newdims.remove(_dim)
                _axes = tuple([np.where(olddims == _dim)[0].item() for _dim in _dims])
                
            else:
                _dims = cast(Tuple[int], _dims)
                for _dim in _dims: newdims.remove(olddims[_dim])
                _axes = _dims
        else:
            newdims = None
            _dims = cast(Tuple[int], _dims)
            _axes = _dims
            _x = x

        y = _np_func(_x, axis=_axes) 
        return DataArray(y, dims=newdims)
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





