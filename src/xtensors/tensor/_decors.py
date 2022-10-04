from __future__ import annotations
from functools import wraps
import numpy.typing as npt

from .broadcast._broadcast import vanilla_broadcaster, cast
from .basic_utils import mergedims, mergecoords


from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from typing import Callable, TypeVar
    from typing_extensions import ParamSpec
    from .typing import BinaryOperator, TernaryOperator
    from .broadcast._types import Broadcaster
    from ._base import XTensor
    O = ParamSpec('O')
    T = TypeVar('T')


def promote_binary_operator(
        broadcaster: Broadcaster|None=None, 
        # dimcoord_converter: Callable[[Dims, Coords], Tuple[Dims, Coords]]|None=None,
    ) -> Callable[
            [BinaryOperator[npt.NDArray]], BinaryOperator[XTensor]
        ]:
    '''
    Promote an NDArray binary operator to XTensor binary operator
    '''
    if broadcaster is None: broadcaster = vanilla_broadcaster
    def wrapper(f: BinaryOperator[npt.NDArray]) -> BinaryOperator[XTensor]:
        @wraps(f)
        def wrapped(X: XTensor, Y: XTensor) -> XTensor:
            from ._base import XTensor
            
            _x, _y, dims, coords = broadcaster(X, Y)
            res_data = f(_x, _y)

            # if dimcoord_converter:
            #     dims, coords = dimcoord_converter(dims, coords)
            return XTensor(res_data, dims, coords)
        return wrapped
    return wrapper


def promote_ternary_operator(
    ) -> Callable[[TernaryOperator[npt.NDArray]], TernaryOperator[XTensor]]:
    '''
    Promote an NDArray ternary operator to XTensor ternary operator using
    vanilla broadcaster
    '''
    broadcaster = vanilla_broadcaster
    def wrapper(f: TernaryOperator[npt.NDArray]) -> TernaryOperator[XTensor]:
        @wraps(f)
        def wrapped(X: XTensor, Y: XTensor, Z: XTensor) -> XTensor:
            from ._base import XTensor

            largest = X
            if X.rank < Y.rank: largest = Y
            if largest.rank < Z.rank: largest = Z

            _cast = cast(broadcaster, largest)

            X1 = _cast(X)
            Y1 = _cast(Y)
            Z1 = _cast(Z)
            res_data = f(X1.data, Y1.data, Z1.data)

            dims = mergedims(mergedims(X, Y), Z)
            coords = mergecoords(mergecoords(X, Y), Z)

            return XTensor(res_data, dims, coords)
        return wrapped
    return wrapper
