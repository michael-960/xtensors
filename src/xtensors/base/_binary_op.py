from typing import Optional, Protocol, Tuple, cast

from xtensors.typing import NDArray
from xarray import DataArray

from ._broadcast import are_shapes_broadcastable, broadcast_xarrays, broadcast_arrays


class BinaryOperation(Protocol):
    def __call__(self, x: NDArray, y: NDArray) -> NDArray: ...


def _apply_operation(x: NDArray, y: NDArray, binop: str, rbinop: Optional[str]=None) -> NDArray:
    try:
        z = getattr(x, binop)(y)
        if z is NotImplemented: raise NotImplementedError

    except AttributeError | NotImplementedError as e:
        if rbinop is not None:
            z = getattr(y, rbinop)(x)
        else:
            raise AttributeError(e) 
        
    return z


def _binop_factory(_bin_op: str, _rbin_op: str) -> BinaryOperation:
    def _op(x: NDArray, y: NDArray) -> NDArray:

        _x, _y, dims, _, = broadcast_arrays(x, y)
        _z = _apply_operation(_x, _y, _bin_op, _rbin_op)

        return DataArray(_z, dims=dims)
    return _op

import numpy as np


_add = _binop_factory('__add__', '__radd__')
_divide = _binop_factory('__truediv__', '__rtruediv__')

_greater = _binop_factory('__gt__', '__lt__')
_greater_equal = _binop_factory('__ge__', '__le__')

_less = _binop_factory('__lt__', '__gt__')
_less_equal = _binop_factory('__le__', '__ge__')

_equal = _binop_factory('__eq__', '__eq__')




