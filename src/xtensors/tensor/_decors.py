from __future__ import annotations
from functools import wraps
import numpy as np
from typing import Callable, Sequence, Tuple 
from typing_extensions import ParamSpec

from ._base import XTensor
from .typing import BinaryOperator, Array, Function_1Arg, Function_2Args, Function_3Args, Dims, Coords

from .broadcast._types import Broadcaster

from .broadcast._broadcast import broadcaster_wrapper
from .broadcast._dimcast import unilateral_dimcast

from .basic_utils import mergedims, mergecoords, dimslast, dimsfirst


O = ParamSpec('O')

def with_broadcast(broadcaster: Broadcaster|None=None, 
    dimcoord_converter: Callable[[Dims, Coords], Tuple[Dims, Coords]]|None=None,
    ) -> Callable[[BinaryOperator[np.ndarray, O]], BinaryOperator[XTensor, O]]:
    '''
    Used to decorate binary operation functions written for numpy NDArrays to
    handle named tensor broadcasting.
    '''

    if broadcaster is None:
        broadcaster = broadcaster_wrapper(
            dimcast=unilateral_dimcast(strict=False),
            dimmerge=mergedims, coordmerge=mergecoords
        )

    def wrapper(f: BinaryOperator[np.ndarray, O]) -> BinaryOperator[XTensor, O]:
        @wraps(f)
        def wrapped(X: XTensor, Y: XTensor, *args: O.args, **kwargs: O.kwargs) -> XTensor:
            _x, _y, dims, coords = broadcaster(X, Y)

            res_data = f(_x, _y, *args, **kwargs)

            if dimcoord_converter:
                dims, coords = dimcoord_converter(dims, coords)
            return XTensor(res_data, dims, coords)

        return wrapped
    return wrapper



def to_xtensor(x: XTensor|Array):
    return x if isinstance(x, XTensor) else XTensor(x.__array__())


def generalize_1(func: Function_1Arg[XTensor, O, XTensor]) -> Function_1Arg[XTensor|Array, O, XTensor]:
    @wraps(func)
    def wrapped(X: Array|XTensor, /, *args: O.args, **kwargs: O.kwargs) -> XTensor:
        return func(to_xtensor(X), *args, **kwargs)
    return wrapped

def generalize_2(func: Function_2Args[XTensor, O, XTensor]) -> Function_2Args[XTensor|Array, O, XTensor]:
    @wraps(func)
    def wrapped(X: Array|XTensor, Y: Array|XTensor, /, *args: O.args, **kwargs: O.kwargs) -> XTensor:
        return func(to_xtensor(X), to_xtensor(Y), *args, **kwargs)
    return wrapped

def generalize_3(func: Function_3Args[XTensor, O, XTensor]) -> Function_3Args[XTensor|Array, O, XTensor]:
    @wraps(func)
    def wrapped(X: Array|XTensor, Y: Array|XTensor, Z: Array|XTensor, /, *args: O.args, **kwargs: O.kwargs) -> XTensor:
        return func(to_xtensor(X), to_xtensor(Y), to_xtensor(Z), *args, **kwargs)
    return wrapped

