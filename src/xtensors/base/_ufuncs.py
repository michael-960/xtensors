from __future__ import annotations
from functools import wraps
from typing import Callable, Protocol
from textwrap import dedent


import numpy as np

from .. import tensor as xtt

from ..tensor import XTensor, TensorLike



class _ufunc(Protocol):
    def __call__(self, __x1: np.ndarray, /, *args) -> np.ndarray: ...


class UFunc(Protocol):
    def __call__(self, x: xtt.TensorLike, /) -> xtt.XTensor: ...


def _ufunc_factory(_np_func: _ufunc) -> UFunc:
    @xtt.generalize_at_0
    def _f(X: xtt.XTensor, /) -> xtt.XTensor:
        return xtt.XTensor(_np_func(X.data), X.dims, X.coords)
    return _f


def _inject_docs(ufunc: UFunc, func_name: str) -> UFunc:
    _docs = dedent(f"""
        :param x: Input tensor

        :return: :math:`{func_name}(x)`

    """)
    ufunc.__doc__ = _docs
    return ufunc


def _inject_sig(ufunc: UFunc) -> UFunc:
    def _dummy(x: TensorLike, /) -> XTensor:
        ...
    _dummy.__doc__ = ufunc.__doc__

    return wraps(_dummy)(ufunc)


def _sigmoid(__x1: np.ndarray):
    return .5 * (1. + np.tanh(__x1/2))


def postproc(*args):
    def _postproc(f: UFunc):
        return _inject_sig(_inject_docs(f, *args))

    return _postproc


# sigmoid = _inject_signature(_inject_docs(_ufunc_factory(_sigmoid), r'\mathrm{sigmoid}'))

sigmoid = postproc(r'\mathrm{sigmoid}')(_ufunc_factory(_sigmoid))

exp = postproc(r'\exp')(_ufunc_factory(np.exp))

log2 = postproc(r'\log_2')(_ufunc_factory(np.log2))

log = postproc(r'\ln')(_ufunc_factory(np.log))

log10 = postproc(r'\log_{10}')(_ufunc_factory(np.log10))

cos = postproc(r'\cos')(_ufunc_factory(np.cos))

sin = postproc(r'\sin')(_ufunc_factory(np.sin))

tan = postproc(r'\tan')(_ufunc_factory(np.tan))

cosh = postproc(r'\cosh')(_ufunc_factory(np.cosh))

sinh = postproc(r'\sinh')(_ufunc_factory(np.sinh))

tanh = postproc(r'\tanh')(_ufunc_factory(np.tanh))





