from __future__ import annotations
import typing


from xarray import DataArray
from xtensors.typing import NDArray


class Functional:
    def __init__(self):
        self.name: str = 'NOTIMPLEMENTED_FUNCTIONAL'

    def __call__(self, x: NDArray) -> DataArray:
        raise NotImplementedError


class Identity(Functional):
    def __init__(self, name='I'):
        self.name = name

    def __call__(self, x: NDArray) -> DataArray:
        if not isinstance(x, DataArray):
            return DataArray(x)

        return x


class Pipe(Functional):
    '''
        Pipe(f1, f2, ..., fn)(x) = fn(...f2(f1(x))...)
    '''
    def __init__(self, *f: Functional, delim: str='.'):
        self.f = f
        self.delim = delim
        self.name = delim.join([_f.name for _f in self.f])

    def __call__(self, x: NDArray) -> DataArray:
        _y = x
        for _f in self.f:
            _y = _f(_y)

        if not isinstance(_y, DataArray):
            _y = DataArray(_y)
        return _y
