from __future__ import annotations
'''
Generalization:
    These decorators take XTensor-only functions and make them compatible with
    any object that supports the __array__() protocol, which is equivalent to
    an XTensor with no named dimensions.

'''
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from ._base import to_xtensor

from ._misc import copy_doc



if TYPE_CHECKING:

    from typing_extensions import ParamSpec, Concatenate
    from .._base import XTensor
    from ..typing import TensorLike

    O = ParamSpec('O')

    T = TypeVar('T')
    U = TypeVar('U')
    V = TypeVar('V')
    R = TypeVar('R')


def generalize_at_0(
        func: Callable[Concatenate[XTensor, O], R]
    ) -> Callable[Concatenate[TensorLike, O], R]:
    '''
    Generalize the first argument of a function from accepting XTensor to
    TensorLike objects
    '''

    @wraps(func)
    def wrapped(X: TensorLike, *args: O.args, **kwargs: O.kwargs) -> R:
        return func(to_xtensor(X), *args, **kwargs)

    return wrapped


def generalize_at_1(
        func: Callable[Concatenate[T, XTensor, O], R]
    ) -> Callable[Concatenate[T, TensorLike, O], R]:
    # Function_2Args[T, TensorLike, O, R]:
    '''Generalize the second argument of a function'''
    @wraps(func)
    def wrapped(arg0: T, X: TensorLike, *args: O.args, **kwargs: O.kwargs) -> R:
        return func(arg0, to_xtensor(X), *args, **kwargs)
    return wrapped


def generalize_at_2(
        func: Callable[Concatenate[T, U, XTensor, O], R]
    ) -> Callable[Concatenate[T, U, TensorLike, O], R]:
    '''Generalize the third argument of a function'''
    @wraps(func)
    def wrapped(arg0: T, arg1: U, X: TensorLike, *args: O.args, **kwargs: O.kwargs) -> R:
        return func(arg0, arg1, to_xtensor(X), *args, **kwargs)
    return wrapped


def generalize_at_3(
        func: Callable[Concatenate[T, U, V, XTensor, O], R]
    ) -> Callable[Concatenate[T, U, V, TensorLike, O], R]:
    '''Generalize the fourth argument of a function'''
    @wraps(func)
    def wrapped(arg0: T, arg1: U, arg2: V, X: TensorLike, *args: O.args, **kwargs: O.kwargs) -> R:
        return func(arg0, arg1, arg2, to_xtensor(X), *args, **kwargs)
    return wrapped

