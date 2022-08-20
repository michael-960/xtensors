from __future__ import annotations
import typing
from numpy.typing import ArrayLike
from xarray import DataArray


class Functional:
    def __init__(self):
        self.name : str = 'NOTIMPLEMENTED_FUNCTIONAL'

    def __call__(self, x: ArrayLike) -> DataArray:
        raise NotImplementedError


class Pipe(Functional):
    '''
        Pipe(f1, f2, ..., fn)(x) = fn(...f2(f1(x))...)
    '''
    def __init__(self, *f: Functional, delim: str='.'):
        assert len(f) > 0
        self.f = f
        self.delim = delim

        self.name = delim.join([_f.name for _f in self.f])

    def __call__(self, x: ArrayLike) -> DataArray:
        _y = x
        for _f in self.f:
            _y = _f(_y)
        assert isinstance(_y, DataArray)
        return _y
