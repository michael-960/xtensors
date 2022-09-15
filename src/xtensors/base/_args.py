from __future__ import annotations
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .. import numpy as xtnp
from .. import tensor as xtt

'''
Args functions:
    For reduction functions whose return values are
    **indices** or **coordinates** over multiple axes
    E.g. argsmax
'''


ARGS_DIM = 'ARGS_DIM'

class _args_func(Protocol):
    def __call__(self, a: np.ndarray, axes: xtnp.AxesLike) -> npt.NDArray[np.int_]: ...

class ArgsFunction(Protocol):
    def __call__(self, x: xtt.TensorLike, /, dims: xtt.DimsLike) -> xtt.XTensor: ...

class CoordsFunction(Protocol):
    def __call__(self, x: xtt.TensorLike, /, dims: xtt.DimsLike, *, 
            use_index_if_no_coord: bool=False) -> xtt.XTensor: ...


def _reduction_factory(_func: _args_func) -> ArgsFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dims: xtt.DimsLike) -> xtt.XTensor:
        '''
        Return
        '''

        axes = X.get_axes(dims)
        r_dims, s_dims = xtt.strip(X.dims, axes, only_remaining=False)
        r_coords, s_coords = xtt.strip(X.coords, axes, only_remaining=False)

        new_dims = r_dims + [ARGS_DIM]
        new_coords = r_coords + [[dim for dim in s_dims]]

        args = _func(X.data, axes=axes)
        return xtt.XTensor(args, dims=new_dims, coords=new_coords)
    return _reduce


argsmin = _reduction_factory(xtnp.argsmin)
argsmax = _reduction_factory(xtnp.argsmax)
nanargsmin = _reduction_factory(xtnp.nanargsmin)
nanargsmax = _reduction_factory(xtnp.nanargsmax)



def _coord_reduc_factory(_func: ArgsFunction) -> CoordsFunction:
    @xtt.generalize_at_0
    def _reduce(X: xtt.XTensor, /, dims: xtt.DimsLike, *, use_index_if_no_coord: bool=False) -> xtt.XTensor:
        '''
        Return the coordinate on [dim] that maximizes/minimizes x If dimension [dim] does
        not have coordinates, this is equivalent to (nan)argmax/argmin
        '''
        args = _func(X, dims)
        axes = X.get_axes(dims)

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


coordsmin = _coord_reduc_factory(argsmin)
coordsmax = _coord_reduc_factory(argsmax)
nancoordsmin = _coord_reduc_factory(nanargsmin)
nancoordsmax = _coord_reduc_factory(nanargsmax)


