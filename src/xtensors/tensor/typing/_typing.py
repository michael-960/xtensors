from __future__ import annotations
from numbers import Real
from typing import Any, Callable, Generic, List, Protocol, Sequence, Tuple, TypeVar, Union
from typing_extensions import ParamSpec, TypeVarTuple

import numpy as np
import numpy.typing as npt

T_con = TypeVar('T_con', contravariant=True)
U_con = TypeVar('U_con', contravariant=True)
V_con = TypeVar('V_con', contravariant=True)
T_co = TypeVar('T_co', covariant=True)

S = TypeVar('S')
T = TypeVar('T')
O = ParamSpec('O')

class Array(Protocol):
    def __array__(self) -> np.ndarray: ...


TensorLike = Union[Array, List, float]

AxesPermutation = List[Union[int, None]]
'''
A list of integers or None that represents a permutation of a set of axes. None
means that a new axis of length 1 is to be created at the corresponding index.
'''

DimLike = Union[str,int,Tuple[str,int]]
DimsLike = List[DimLike]

Dims = List[Union[str,None]]
Coords = List[Union[npt.NDArray[Any],None]]


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











