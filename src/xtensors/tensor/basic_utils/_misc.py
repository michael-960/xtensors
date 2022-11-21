from __future__ import annotations
from functools import wraps


from typing import Callable, List, Literal, Sequence, Tuple, TypeVar, overload


T = TypeVar('T')

@overload
def strip(l: Sequence[T], indices: Sequence[int]) -> List[T]: ...
@overload
def strip(l: Sequence[T], indices: Sequence[int], only_remaining: Literal[True]) -> List[T]: ...
@overload
def strip(l: Sequence[T], indices: Sequence[int], only_remaining: Literal[False]) -> Tuple[List[T], List[T]]: ...

def strip(l: Sequence[T], indices: Sequence[int], only_remaining: bool=True) -> List[T]|Tuple[List[T], List[T]]:
    remaining, stripped = [], []

    for i, x in enumerate(l):
        if i in indices: stripped.append(x)
        else: remaining.append(x)

    if only_remaining: return remaining
    return remaining, stripped



def copy_sig(f: Callable):
    def wrapper(g: T) -> T:
        _doc = g.__doc__
        g = wraps(f)(g)
        g.__doc__ = _doc
        return g

    return wrapper


def copy_doc(f: Callable):

    def wrapper(g: T) -> T:
        g.__doc__ = f.__doc__
        return g
    return wrapper

