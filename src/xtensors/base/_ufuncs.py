from typing import Protocol
from numpy.typing import ArrayLike

from numpy import typing

import numpy as np
from xarray import DataArray
from xtensors.typing import NDArray
import torch


class _ufunc(Protocol):
    def __call__(self, __x1: np.ndarray, *args) -> np.ndarray: ...


class UFunc(Protocol):
    def __call__(self, x: NDArray) -> DataArray: ...



def _ufunc_factory(_np_func: _ufunc) -> UFunc:
    def _f(x: NDArray) -> DataArray:
        _x = np.array(x)
        _y = _np_func(_x)
        _dims = None
        
        if isinstance(x, DataArray):
            _dims = x.dims
        return DataArray(_y, dims=_dims)
    return _f


def _sigmoid(__x1: np.ndarray):

    return .5 * (1. + np.tanh(__x1/2))




sigmoid = _ufunc_factory(_sigmoid)

exp = _ufunc_factory(np.exp)
log = _ufunc_factory(np.log)
log10 = _ufunc_factory(np.log10)

cos = _ufunc_factory(np.cos)
sin = _ufunc_factory(np.sin)
tan = _ufunc_factory(np.tan)

cosh = _ufunc_factory(np.cosh)
sinh = _ufunc_factory(np.sinh)
tanh = _ufunc_factory(np.tanh)
