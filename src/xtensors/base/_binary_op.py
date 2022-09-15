from __future__ import annotations

from .. import tensor as xtt

from typing import Optional, Protocol

import warnings


class BinaryOperation(Protocol):
    def __call__(self, x: xtt.TensorLike, y: xtt.TensorLike, /) -> xtt.XTensor: ...


def _apply_operation(X: xtt.XTensor, Y: xtt.XTensor, binop: str, rbinop: Optional[str]=None) -> xtt.XTensor:

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
        try:
            Z = getattr(X, binop)(Y)
            if Z is NotImplemented: raise NotImplementedError

        except (AttributeError, NotImplementedError) as e:
            if rbinop is not None:
                Z = getattr(Y, rbinop)(X)
                if Z is NotImplemented: raise NotImplementedError
            else:
                raise AttributeError from e
    return Z


def _binop_factory(_bin_op: str, _rbin_op: str) -> BinaryOperation:
    @xtt.generalize_at_1
    @xtt.generalize_at_0
    def _op(X: xtt.XTensor, Y: xtt.XTensor, /) -> xtt.XTensor:
        return _apply_operation(X, Y, _bin_op, _rbin_op)

    return _op


_add = _binop_factory('__add__', '__radd__')
_divide = _binop_factory('__truediv__', '__rtruediv__')
_multiply = _binop_factory('__mul__', '__rmul__')

_greater = _binop_factory('__gt__', '__lt__')
_greater_equal = _binop_factory('__ge__', '__le__')

_less = _binop_factory('__lt__', '__gt__')
_less_equal = _binop_factory('__le__', '__ge__')

_equal = _binop_factory('__eq__', '__eq__')


