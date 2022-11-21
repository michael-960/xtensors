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


from ._args import ArgsFunction, argsmin, argsmax, nanargsmin, nanargsmax


class CoordsFunction(Protocol):
    """
    Coordinate function protocol
    """
    def __call__(self, x: xtt.TensorLike, /, dim: xtt.DimsLike, *, 
            use_index_if_no_coord: bool=False) -> xtt.XTensor:
        """
        :param x: target tensor
        :param dim: target dimension(s)
        :param use_index_if_no_coord: whether to use indices instead of
                        coordinates are not specified

        :return: an :py:class:`XTensor` of rank :code:`x.rank - len(dim) + 1`
        """
        ...


def _coord_reduc_factory(_func: ArgsFunction) -> CoordsFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dim: xtt.DimsLike, *, use_index_if_no_coord: bool=False) -> xtt.XTensor:
        '''
        Return the coordinate on [dim] that maximizes/minimizes x If dimension [dim] does
        not have coordinates, this is equivalent to (nan)argmax/argmin
        '''
        args = _func(X, dim)
        axes = X.get_axes(dim)

        coords = [X.coords[axis] for axis in axes]
        coords_r = []

        for i, (axis, coord) in enumerate(zip(axes, coords)):

            if coord is None: 
                if use_index_if_no_coord:
                    coord_ = np.arange(X.shape[axis])
                else:
                    raise ValueError(f'Tensor has no coordinates on axis {axis}')
            else:
                coord_ = coord


            coords_r.append(coord_[args.data[...,i]])

        return xtt.XTensor(
                np.stack(coords_r, axis=-1),
                dims=args.dims, 
                coords=args.coords)

    return _reduce

def _inject_docs(r: CoordsFunction, doc: str):
    r.__doc__ = dedent(doc)
    return r

def _inject_sig(r: CoordsFunction):
    def _dummy(x: TensorLike, /, dim: DimLike|DimsLike|None, *,
                use_index_if_no_coord: bool=False
        ) -> XTensor:
        ...
    _dummy.__doc__ = r.__doc__
    return wraps(_dummy)(r)

def postproc(doc: str):
    def _postproc(r: CoordsFunction):
        return _inject_sig(_inject_docs(r, doc))

    return _postproc


coordsmin = postproc("""
                     Return the coordinates where minima occur
                     """)(_coord_reduc_factory(argsmin))

coordsmax = postproc("""
                     Return the coordinates where maxima occur
                     """)(_coord_reduc_factory(argsmax))

nancoordsmin = postproc("""
                        Similar to :py:func:`coordsmin`, but with :code:`nan` ignored
                        """)(_coord_reduc_factory(nanargsmin))

nancoordsmax = postproc("""
                        Similar to :py:func:`coordsmax`, but with :code:`nan` ignored
                        """)(_coord_reduc_factory(nanargsmax))


