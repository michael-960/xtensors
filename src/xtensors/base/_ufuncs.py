from __future__ import annotations
from typing import Protocol

import numpy as np

from .. import tensor as xtt


class _ufunc(Protocol):
    def __call__(self, __x1: np.ndarray, /, *args) -> np.ndarray: ...


class UFunc(Protocol):
    def __call__(self, x: xtt.TensorLike, /) -> xtt.XTensor: ...


def _ufunc_factory(_np_func: _ufunc) -> UFunc:
    @xtt.generalize_at_0
    def _f(X: xtt.XTensor, /) -> xtt.XTensor:
        return xtt.XTensor(_np_func(X.data), X.dims, X.coords)
    return _f


def _sigmoid(__x1: np.ndarray):
    return .5 * (1. + np.tanh(__x1/2))

sigmoid = _ufunc_factory(_sigmoid)

exp = _ufunc_factory(np.exp)

log2 = _ufunc_factory(np.log2)
log = _ufunc_factory(np.log)
log10 = _ufunc_factory(np.log10)

cos = _ufunc_factory(np.cos)
sin = _ufunc_factory(np.sin)
tan = _ufunc_factory(np.tan)

cosh = _ufunc_factory(np.cosh)
sinh = _ufunc_factory(np.sinh)
tanh = _ufunc_factory(np.tanh)

