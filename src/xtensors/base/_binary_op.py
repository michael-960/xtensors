from __future__ import annotations
from functools import wraps
from textwrap import dedent

from .. import tensor as xtt

from ..tensor import XTensor, TensorLike

from typing import Optional, Protocol
import numpy as np
import numpy.typing as npt

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


def _inject_docs(b: BinaryOperation, operation: str) -> BinaryOperation:
    
    b.__doc__ = dedent(
        fr"""
            :param x,y: :py:class:`xtensors.TensorLike` objects

            :return: :math:`{operation}`
        """
    )

    return b


def _inject_sig(b: BinaryOperation) -> BinaryOperation:
    def _dummy(x: TensorLike, y: TensorLike, /) -> XTensor:
        ...

    _dummy.__doc__ = b.__doc__
    return wraps(_dummy)(b)


def postproc(operation: str):
    def _postproc(b: BinaryOperation):
        return _inject_sig(_inject_docs(b, operation))

    return _postproc

_add = postproc('x + y')(_binop_factory('__add__', '__radd__'))

_divide = postproc('x / y')(_binop_factory('__truediv__', '__rtruediv__'))

_multiply = postproc('xy')(_binop_factory('__mul__', '__rmul__'))

_greater = postproc('x > y')(_binop_factory('__gt__', '__lt__'))

_greater_equal = postproc(r'x \ge y')(_binop_factory('__ge__', '__le__'))

_less = postproc(r'x < y')(_binop_factory('__lt__', '__gt__'))

_less_equal = postproc(r'x \le y')(_binop_factory('__le__', '__ge__'))

_equal = postproc(r'x = y')(_binop_factory('__eq__', '__eq__'))


def _np_or(X: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
    return np.logical_or(X, Y)

def _np_and(X: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
    return np.logical_and(X, Y)




_or = postproc(r'x\;\mathrm{or}\;y')(
        xtt.generalize_at_0(
        xtt.generalize_at_1(
        xtt.promote_binary_operator(xtt.vanilla_broadcaster)(
            _np_or
))))


_and = postproc(r'x\;\mathrm{and}\;y')(
        xtt.generalize_at_0(
        xtt.generalize_at_1(
        xtt.promote_binary_operator(xtt.vanilla_broadcaster)(
            _np_and
))))


