from __future__ import annotations
from functools import wraps
from textwrap import dedent
'''
Args functions:
    For reduction functions whose return values are
    **indices** or **coordinates** over multiple axes
    E.g. argsmax
'''

from typing import Protocol

import numpy as np
import numpy.typing as npt

from .. import numpy as xtnp
from .. import tensor as xtt

from ..tensor import XTensor, TensorLike, DimLike, DimsLike


ARGS_DIM = 'ARGS_DIM'

class _args_func(Protocol):
    def __call__(self, a: np.ndarray, axes: xtnp.AxesLike) -> npt.NDArray[np.int_]: ...


class ArgsFunction(Protocol):
    """
    Argument function protocol
    """
    def __call__(self, x: TensorLike, /, dim: DimsLike) -> XTensor:
        """
        :param x: target tensor
        :param dim: target dimension(s)

        :return: an :py:class:`XTensor` of rank :code:`x.rank - len(dim) + 1`

        """
        ...


def _reduction_factory(_func: _args_func) -> ArgsFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimsLike) -> xtt.XTensor:
        '''
        Return
        '''

        axes = X.get_axes(dim)
        r_dims, s_dims = xtt.strip(X.dims, axes, only_remaining=False)
        r_coords, s_coords = xtt.strip(X.coords, axes, only_remaining=False)

        new_dims = r_dims + [ARGS_DIM]
        new_coords = r_coords + [[dim for dim in s_dims]]

        args = _func(X.data, axes=axes)
        return xtt.XTensor(args, dims=new_dims, coords=new_coords)
    return _reduce


def _inject_docs(r: ArgsFunction, doc: str):
    r.__doc__ = dedent(doc)
    return r

def _inject_sig(r: ArgsFunction):
    def _dummy(x: TensorLike, /, dim: DimLike|DimsLike|None) -> XTensor:
        ...
    _dummy.__doc__ = r.__doc__
    return wraps(_dummy)(r)

def postproc(doc: str):
    def _postproc(r: ArgsFunction):
        return _inject_sig(_inject_docs(r, doc))

    return _postproc


argsmin = postproc("""
                    Return the indices where minima occur.
                   """)(_reduction_factory(xtnp.argsmin))


argsmax = postproc("""
                    Return the indices where maxima occur.
                    """)(_reduction_factory(xtnp.argsmax))

nanargsmin = postproc("""
                      Similar to :py:func:`argsmin`, but with :code:`nan` ignored
                      """)(_reduction_factory(xtnp.nanargsmin))

nanargsmax = postproc("""
                      Similar to :py:func:`argsmax`, but with :code:`nan` ignored
                      """)(_reduction_factory(xtnp.nanargsmax))



