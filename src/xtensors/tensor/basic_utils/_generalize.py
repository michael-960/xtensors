from __future__ import annotations
'''
Generalization:
    These decorators take XTensor-only functions and make them compatible with
    any object that supports the __array__() protocol, which is equivalent to
    an XTensor with no named dimensions.

'''
from functools import wraps
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

from ._base import to_xtensor


if TYPE_CHECKING:
    from .._base import XTensor
    from ..typing import Array, Function_1Arg, Function_2Args, Function_3Args, TensorLike

O = ParamSpec('O')
T = TypeVar('T')

def generalize_1(func: Function_1Arg[XTensor, O, T]) -> Function_1Arg[TensorLike, O, T]:
    @wraps(func)
    def wrapped(X: TensorLike, /, *args: O.args, **kwargs: O.kwargs) -> T:
        return func(to_xtensor(X), *args, **kwargs)
    return wrapped

def generalize_2(func: Function_2Args[XTensor, O, T]) -> Function_2Args[TensorLike, O, T]:
    @wraps(func)
    def wrapped(X: TensorLike, Y: TensorLike, /, *args: O.args, **kwargs: O.kwargs) -> T:
        return func(to_xtensor(X), to_xtensor(Y), *args, **kwargs)
    return wrapped

def generalize_3(func: Function_3Args[XTensor, O, T]) -> Function_3Args[TensorLike, O, T]:
    @wraps(func)
    def wrapped(X: TensorLike, Y: TensorLike, Z: TensorLike, /, *args: O.args, **kwargs: O.kwargs) -> T:
        return func(to_xtensor(X), to_xtensor(Y), to_xtensor(Z), *args, **kwargs)
    return wrapped
