from __future__ import annotations

'''
Template Broadcaster:

    First specify a template, then map axes of tensors onto the template

'''

from typing import List, Literal, Optional, Sequence, Union
import numpy as np
from ._types import Broadcaster
from .._base import XTensor

from ..typing import Dims, Coords
from ..basic_utils import permute, permutation_well_defined, mergecoords, mergedims


Channel = Union[str, None]


def template_broadcast(dims: Sequence[Union[str, int]], channels: Sequence[Literal['x','y']|None]) -> Broadcaster:

    template = Template.from_dims_channels(dims, channels)
    
    def _broadcast(X: XTensor, Y: XTensor):
        X1 = template.cast_and_update(X, 'x')
        Y1 = template.cast_and_update(Y, 'y')

        dims, coords = template.dims, template.coords
        template.clear()
        return X1.data, Y1.data, dims, coords

    return _broadcast



class AxisSelector:
    def __init__(self) -> None: ...
    def select_axis(self, X: XTensor, channel: str) -> int|None: ...


class DimNameSelector(AxisSelector):
    def __init__(self, dimname: str, 
            required: bool=False, channel: Optional[str]=None
            ) -> None:
        self.dimname = dimname
        self.required = required
        self.channel = channel

    def select_axis(self, X: XTensor, channel: str) -> int|None:
        if self.channel is not None and channel != self.channel:
            return None
        try:
            return X.get_axis(self.dimname)
        except KeyError as e:
            if self.required: raise e
            return None


class IndexSelector(AxisSelector):
    def __init__(self, axis: int, channel: Optional[str]=None) -> None:
        self.axis = axis
        self.channel = channel


    def select_axis(self, X: XTensor, channel: str) -> int | None:
        if self.channel is not None and channel != self.channel:
            return None
        
        if self.axis >= 0:
            return self.axis
        else:
            return self.axis + X.rank


class Template:
    def __init__(self, *selectors: AxisSelector) -> None:
        self.selectors = selectors
        self._dims: Dims
        self._coords: Coords
        self.clear()
        self.check_well_defined()

    @classmethod
    def from_dims_channels(
            cls, dims: Sequence[Union[str, int]], channels: Sequence[str|None]) -> Template:
        selectors = []

        for d, c in zip(dims, channels):
            if isinstance(d, str):
                selectors.append(DimNameSelector(d, channel=c))
            else:
                selectors.append(IndexSelector(d, channel=c))
        
        return cls(*selectors)

    def check_well_defined(self):
        names = [sel.dimname for sel in self.selectors if isinstance(sel, DimNameSelector)]
        assert len(names) == len(set(names))

        axes = [sel.axis for sel in self.selectors if isinstance(sel, IndexSelector)]
        assert len(axes) == len(set(axes))

    def cast_and_update(self, X: XTensor, channel: str) -> XTensor:
        axesp = [sel.select_axis(X, channel) for sel in self.selectors]
        assert permutation_well_defined(axesp)
        print(axesp)

        X1 = permute(X, axesp)

        self._dims = mergedims(self._dims, X1)
        self._coords = mergecoords(self._coords, X1)

        return X1

    def clear(self):
        self._dims = [None for _ in range(len(self.selectors))]
        self._coords= [None for _ in range(len(self.selectors))]

    @property
    def dims(self) -> Dims:
        return self._dims

    @property
    def coords(self) -> Coords:
        return self._coords

