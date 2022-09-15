from __future__ import annotations
from typing import List, Literal, Sequence, Tuple, TypeVar, overload



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




