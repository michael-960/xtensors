
from typing import Any, Protocol, Tuple, cast

from xtensors.typing import NDArray, DimLike
from xtensors.unify import get_axes, strip_dims

from xarray import DataArray

import numpy as np
import numpy.typing as npt

from .. import tensor as xtt

'''
Arg functions:
    For reduction functions whose return values are array **indices** or
    **coordinates**
    E.g. argmax
'''

class _np_arg_func(Protocol):
    def __call__(self, a: np.ndarray, axis: int) -> npt.NDArray[np.int_]: ...

class ArgFunction(Protocol):
    def __call__(self, x: xtt.Array, /, dim: xtt.DimLike) -> xtt.XTensor: ...


def _reduction_factory(_np_func: _np_arg_func) -> ArgFunction:

    @xtt.generalize_1
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimLike) -> xtt.XTensor:
        axis = X.get_axis(dim)
        return xtt.XTensor(
                _np_func(X.data, axis=axis),
                dims=xtt.strip(X.dims, [axis]), 
                coords=xtt.strip(X.coords, [axis]))
    return _reduce

_argmax = _reduction_factory(np.argmax)
_argmin = _reduction_factory(np.argmin)

_nanargmax = _reduction_factory(np.nanargmax)
_nanargmin = _reduction_factory(np.nanargmin)


def _coord_reduc_factory(_func: ArgFunction) -> ArgFunction:

    @xtt.generalize_1
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimLike) -> xtt.XTensor:
        '''
            Return the coordinate on [dim] that maximizes/minimizes x If dimension [dim] does
            not have coordinates, this is equivalent to (nan)argmax/argmin
        '''
        A = _func(X, dim)
        axis = X.get_axis(dim)
        X.coords

        return X.coords[axis][args]
    return _reduce


_coordmax = _coord_reduc_factory(_argmax)
_coordmin = _coord_reduc_factory(_argmin)


_nancoordmax = _coord_reduc_factory(_nanargmax)
_nancoordmin = _coord_reduc_factory(_nanargmin)



