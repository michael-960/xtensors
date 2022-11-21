from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from typing import List, Protocol, Sequence, Tuple, TypeVar, Union, Any
import numpy as np


class Array(Protocol):
    """Anything that implements the :code:`__array__` protocol"""
    def __array__(self) -> np.ndarray: ...


class HasDimName(Protocol):
    """Anything that implements the :code:`__get_dimname__` protocol"""

    def __get_dimname__(self) -> str: ...


TensorLike = Union[Array, Sequence, float]

DimLike = Union[str,int,Tuple[str,int],HasDimName]

DimsLike = List[DimLike]

AxesPermutation = List[Union[int, None]]
'''
A list of integers or None that represents a permutation of a set of axes. None
means that a new axis of length 1 is to be created at the corresponding index.
'''

Dims = List[Union[str,None]]
Coords = List[Union[np.ndarray,None]]


if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    import numpy.typing as npt


    T_con = TypeVar('T_con', contravariant=True)
    U_con = TypeVar('U_con', contravariant=True)
    V_con = TypeVar('V_con', contravariant=True)
    T_co = TypeVar('T_co', covariant=True)

    S = TypeVar('S')
    T = TypeVar('T')
    O = ParamSpec('O')

    
    

    class Function_1Arg(Protocol[T_con, O, T_co]):
        def __call__(self, X: T_con, /, *args: O.args, **kwargs: O.kwargs) -> T_co: ...


    class Function_2Args(Protocol[T_con, U_con, O, T_co]):
        def __call__(self, X: T_con, Y: U_con, /, *args: O.args, **kwargs: O.kwargs) -> T_co: ...


    class Function_3Args(Protocol[T_con, U_con, V_con, O, T_co]):
        def __call__(self, X: T_con, Y: U_con, Z: V_con, /, *args: O.args, **kwargs: O.kwargs) -> T_co: ...


    class BinaryOperator(Protocol[T]):
        def __call__(self, X: T, Y: T) -> T: ...


    class TernaryOperator(Protocol[T]):
        def __call__(self, X: T, Y: T, Z: T) -> T: ...



