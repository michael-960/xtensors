from __future__ import annotations
from functools import wraps
from textwrap import dedent
from typing import Protocol


import numpy as np
import numpy.typing as npt
from .. import tensor as xtt
from ..tensor import XTensor

'''
Arg functions:
    For single-dimension reduction functions whose return values are array
    **indices** or **coordinates**
    E.g. argmax
'''

class _np_arg_func(Protocol):
    def __call__(self, a: np.ndarray, axis: int) -> npt.NDArray[np.int_]: ...

class ArgFunction(Protocol):
    def __call__(self, x: xtt.TensorLike, /, dim: xtt.DimLike) -> xtt.XTensor: ...

class CoordFunction(Protocol):
    def __call__(self, x: xtt.TensorLike, /, dim: xtt.DimLike, *, 
            use_index_if_no_coord: bool=False) -> xtt.XTensor: ...


def _reduction_factory(_np_func: _np_arg_func) -> ArgFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimLike) -> xtt.XTensor:
        axis = X.get_axis(dim)
        return xtt.XTensor(
                _np_func(X.data, axis=axis),
                dims=xtt.strip(X.dims, [axis]),
                coords=xtt.strip(X.coords, [axis]))
    return _reduce


def _inject_docs(r: ArgFunction, doc: str):
    r.__doc__ = dedent(doc)
    return r

def _inject_sig(r: ArgFunction):
    def _dummy(x: xtt.TensorLike, /, dim: xtt.DimLike|xtt.DimsLike|None) -> XTensor:
        ...
    _dummy.__doc__ = r.__doc__
    return wraps(_dummy)(r)

def postproc(doc: str):
    def _postproc(r: ArgFunction):
        return _inject_sig(_inject_docs(r, doc))

    return _postproc


_argmax = postproc("""
                    :return: 
                   """)(_reduction_factory(np.argmax))

_argmin = _reduction_factory(np.argmin)

_nanargmax = _reduction_factory(np.nanargmax)
_nanargmin = _reduction_factory(np.nanargmin)


def _coord_reduc_factory(_func: ArgFunction) -> CoordFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimLike, *, use_index_if_no_coord: bool=False) -> xtt.XTensor:
        '''
            Return the coordinate on [dim] that maximizes/minimizes x If dimension [dim] does
            not have coordinates, this is equivalent to (nan)argmax/argmin
        '''
        args = _func(X, dim)
        axis = X.get_axis(dim)
        coord = X.coords[axis]

        if coord is None: 
            if use_index_if_no_coord:
                coord_r = np.arange(X.shape[axis])[args]
            else:
                raise ValueError(f'Tensor has no coordinates on axis {axis}')
        else:
            coord_r = coord[args]

        return xtt.XTensor(coord_r, dims=xtt.strip(X.dims, [axis]), coords=xtt.strip(X.coords, [axis]))
    return _reduce


_coordmax = _coord_reduc_factory(_argmax)
_coordmin = _coord_reduc_factory(_argmin)

_nancoordmax = _coord_reduc_factory(_nanargmax)
_nancoordmin = _coord_reduc_factory(_nanargmin)


