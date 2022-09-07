from __future__ import annotations
from typing import Optional, Protocol, Tuple, cast

from xtensors.typing import NDArray
from xarray import DataArray

from ._broadcast import are_shapes_broadcastable, broadcast_xarrays, broadcast_arrays

from ..unify import get_coord

import xarray as xr
import warnings


class BinaryOperation(Protocol):
    def __call__(self, x: NDArray, y: NDArray) -> DataArray: ...


def _apply_operation(x: NDArray, y: NDArray, binop: str, rbinop: Optional[str]=None) -> NDArray:

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
        try:
            z = getattr(x, binop)(y)
            if z is NotImplemented: raise NotImplementedError

        except (AttributeError, NotImplementedError) as e:
            if rbinop is not None:
                z = getattr(y, rbinop)(x)
                if z is NotImplemented: raise NotImplementedError
            else:
                raise AttributeError(e) 
        
    return z


def _binop_factory(_bin_op: str, _rbin_op: str) -> BinaryOperation:
    def _op(x: NDArray, y: NDArray) -> DataArray:

        _x, _y, dims, _, = broadcast_arrays(x, y)
        _z = _apply_operation(_x, _y, _bin_op, _rbin_op)

        coords_map = dict()
        if dims is not None:
            for dimkey in dims:
                coord_x = get_coord(_x, dimkey)
                coord_y = get_coord(_y, dimkey)

                if coord_x is not None: coords_map[dimkey] = coord_x
                elif coord_y is not None: coords_map[dimkey] = coord_y

        return DataArray(_z, dims=dims, coords=coords_map)
    return _op

import numpy as np

_add = _binop_factory('__add__', '__radd__')
_divide = _binop_factory('__truediv__', '__rtruediv__')
_multiply = _binop_factory('__mul__', '__rmul__')

_greater = _binop_factory('__gt__', '__lt__')
_greater_equal = _binop_factory('__ge__', '__le__')

_less = _binop_factory('__lt__', '__gt__')
_less_equal = _binop_factory('__le__', '__ge__')

_equal = _binop_factory('__eq__', '__eq__')




