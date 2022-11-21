from __future__ import annotations
from functools import wraps
from textwrap import dedent
from typing import Protocol, Tuple
import numpy as np

from .. import tensor as xtt
from ..tensor import XTensor, TensorLike, DimLike, DimsLike


'''
For reduction functions that can act over multiple axes/dims.
'''

class _np_reduction_func(Protocol):
    def __call__(self, a: np.ndarray, axis: int|Tuple[int,...]) -> np.ndarray: ...


class ReductionFunc(Protocol):
    def __call__(self, x: xtt.TensorLike,/, dim: xtt.DimLike|xtt.DimsLike|None) -> xtt.XTensor: ...


def _reduction_factory(_np_func: _np_reduction_func) -> ReductionFunc:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimLike|xtt.DimsLike|None=None) -> xtt.XTensor:
        axes = X.get_axes(dim)

        _y = _np_func(X.data, axis=tuple(axes))

        return xtt.XTensor(_y, dims=xtt.strip(X.dims, axes), coords=xtt.strip(X.coords, axes))
    return _reduce


def _inject_docs(r: ReductionFunc, doc: str):
    r.__doc__ = dedent(doc)
    return r

def _inject_sig(r: ReductionFunc):
    def _dummy(x: TensorLike, /, dim: DimLike|DimsLike|None) -> XTensor:
        ...

    _dummy.__doc__ = r.__doc__
    return wraps(_dummy)(r)

def postproc(doc: str):
    def _postproc(r: ReductionFunc):
        return _inject_sig(_inject_docs(r, doc))

    return _postproc



_sum = postproc(r"""
                :return: :math:`\sum_{\mathrm{dim}} x`

                """)(_reduction_factory(np.sum))

_mean = postproc(r"""
                :return: :math:`\braket{x}_\mathrm{dim}`
                 """)(_reduction_factory(np.mean))


_std = postproc(r"""
                :return: :math:`\sqrt{\braket{x^2}_\mathrm{dim} - \braket{x}_\mathrm{dim}^2}`
                """)(_reduction_factory(np.std))

_nanmean = postproc(r"""
                    Same as :py:meth:`xtensors.mean`, but :code:`nan` is ignored
                    """)(_reduction_factory(np.nanmean))

_nanstd = postproc(r"""
                   Same as :py:meth:`xtensors.std`, but :code:`nan` is ignored
                    """)(_reduction_factory(np.nanstd))

_nansum = postproc(r"""
                   Same as :py:meth:`xtensors.sum`, but :code:`nan` is ignored
                    """)(_reduction_factory(np.nansum))


_max = postproc(r"""
                :return: :math:`\max_\mathrm{dim} x`
                """)(_reduction_factory(np.max))

_min = postproc(r"""
                :return: :math:`\min_\mathrm{dim} x`
                """)(_reduction_factory(np.min))

_nanmax = postproc(r"""
                Same as :py:meth:`xtensors.max`, but :code:`nan` is ignored
                """)(_reduction_factory(np.nanmax))

_nanmin = postproc(r"""
                Same as :py:meth:`xtensors.min`, but :code:`nan` is ignored
                """)(_reduction_factory(np.nanmin))

_all = postproc(r"""
                Used on bool-valued :code:`TensorLike` objects,
                similar to :code:`np.all`
                """)(_reduction_factory(np.all))


_any = postproc(r"""Used on bool-valued :code:`TensorLike` objects,
                similar to :code:`np.any`
                """)(_reduction_factory(np.any))





