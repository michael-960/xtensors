
from typing import Any, Protocol, Tuple, cast

from xtensors.typing import NDArray, DimLike
from xtensors.unify import get_axes, strip_dims

from xarray import DataArray

import numpy as np

'''
For reduction functions that only make sense when acting over a single axis
E.g. argmax
'''


class _np_single_reduction_func(Protocol):
    def __call__(self, a: NDArray, axis: int) -> DataArray: ...

class SingleDimReductionFunc(Protocol):
    def __call__(self, x: NDArray, dim: DimLike) -> DataArray: ...


def _reduction_factory(_np_func: _np_single_reduction_func) -> SingleDimReductionFunc:
    def _reduce(x: NDArray, dim: DimLike) -> DataArray:
        axis = get_axes(x, dim)[0]

        _y = _np_func(x.__array__(), axis=axis)

        dims = None
        if isinstance(x, DataArray):
            _dims = cast(Tuple[str,...], x.dims)
            dims = strip_dims(_dims, (axis,))

        return DataArray(_y, dims=dims)


    return _reduce


_argmax = _reduction_factory(np.argmax)
_argmin = _reduction_factory(np.argmin)

_nanargmax = _reduction_factory(np.nanargmax)
_nanargmin = _reduction_factory(np.nanargmin)

