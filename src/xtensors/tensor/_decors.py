from __future__ import annotations
from functools import wraps
import numpy as np
from typing import TYPE_CHECKING, Callable, Sequence, Tuple 
from typing_extensions import ParamSpec


from .typing import BinaryOperator, Array, Function_1Arg, Function_2Args, Function_3Args, Dims, Coords

from .broadcast._types import Broadcaster

from .broadcast._broadcast import broadcaster_wrapper
from .broadcast._dimcast import unilateral_dimcast

from .basic_utils import mergedims, mergecoords, dimslast, dimsfirst, to_xtensor

if TYPE_CHECKING:
    from ._base import XTensor


O = ParamSpec('O')

def promote_binary_operator(broadcaster: Broadcaster|None=None, 
    dimcoord_converter: Callable[[Dims, Coords], Tuple[Dims, Coords]]|None=None,
    ) -> Callable[[BinaryOperator[np.ndarray]], BinaryOperator[XTensor]]:
    '''
    Promote NDArray binary operators to XTensor binary operator
    '''

    if broadcaster is None:
        broadcaster = broadcaster_wrapper(
            dimcast=unilateral_dimcast(strict=False),
            dimmerge=mergedims, coordmerge=mergecoords
        )

    def wrapper(f: BinaryOperator[np.ndarray]) -> BinaryOperator[XTensor]:
        @wraps(f)
        def wrapped(X: XTensor, Y: XTensor, /) -> XTensor:
            from ._base import XTensor
            _x, _y, dims, coords = broadcaster(X, Y)

            res_data = f(_x, _y)

            if dimcoord_converter:
                dims, coords = dimcoord_converter(dims, coords)
            return XTensor(res_data, dims, coords)
        return wrapped
    return wrapper



