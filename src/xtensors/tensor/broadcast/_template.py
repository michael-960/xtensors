from __future__ import annotations
from abc import abstractmethod

'''
Template Broadcaster:

    First specify a template, then map axes of tensors onto the template

'''

import numpy as np
from ..basic_utils import permute, permutation_well_defined, mergecoords, mergedims

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Literal, Optional, Sequence, Union
    from .._base import XTensor
    from ..typing import Dims, Coords

    Channel = Union[str, None]


class AxisSelector:
    """
    Abstract base class for objects that implement :code:`select_axis`
    
    """
    def __init__(self) -> None:
        self._label: str
        self.channel: str|None

    @abstractmethod
    def select_axis(self, X: XTensor, channel: str|None=None) -> int|None:
        """
        Select an axis from the tensor X
        
        :param X: tensor
        :param channel: (optional) a string specifying the channel. If
                :code:`self.channel` is set, 
                axis selection will only be performed
                when the :code:`channel` parameter
                matches :code:`self.channel`, or else :code:`None` is returned.
        """

    @property
    def label(self) -> str:
        return self._label


class DimNameSelector(AxisSelector):
    """
    Select an axis based on its name

    """
    def __init__(
        self, dimname: str, 
        required: bool=False, channel: Optional[str]=None) -> None:
        """
        :param dimname: target axis name
        :param required: if :code:`True`, a :code:`KeyError` will be raised if 
                no axis named :code:`dimname` is found when selecting an axis from a tensor.

        :param channel: Selecting channel.

        """
        self.dimname = dimname
        self.required = required
        self.channel = channel
        self._label = dimname

    def select_axis(self, X: XTensor, channel: str|None=None) -> int|None:
        if self.channel is not None and channel != self.channel:
            return None
        try:
            return X.get_axis(self.dimname)
        except KeyError as e:
            if self.required: raise e
            return None



class IndexSelector(AxisSelector):
    """
    Select an axis based on the index

    """
    def __init__(self, axis: int, channel: Optional[str]=None) -> None:
        """
        :param axis: The axis index, negative indices are supported
        :param channel: Selecting channel.

        """
        self.axis = axis
        self.channel = channel
        self._label = str(axis)


    def select_axis(self, X: XTensor, channel: str|None=None) -> int | None:
        if self.channel is not None and channel != self.channel:
            return None
        
        if self.axis >= 0:
            return self.axis
        else:
            return self.axis + X.rank


class Template:
    """
    A tensor broadcasting template that can be reused to broadcast multiple
    tensors.

    A template is basically a list of :py:class:`AxisSelector` instances, each
    specifying how an axis (integer) should be selected.

    """
    def __init__(self, *selectors: AxisSelector) -> None:
        """
        :param selectors: A list of :py:class:`AxisSelector`
        """

        try:
            for sel in selectors: assert isinstance(sel, AxisSelector)
        except AssertionError as e:
            raise TypeError('selectors must be instances of AxisSelector') from e

        self.selectors = selectors
        self._dims: Dims
        self._coords: Coords
        self.clear()
        self.check_well_defined()


    @classmethod
    def from_dims_channels(
            cls,
            dims: Sequence[Union[str, int]],
            channels: Sequence[str|None]|None=None) -> Template:
        selectors = []

        if channels is None: channels = [None for _ in range(len(dims))]

        assert len(dims) == len(channels)

        for d, c in zip(dims, channels):
            if isinstance(d, str):
                selectors.append(DimNameSelector(d, channel=c))
            else:
                selectors.append(IndexSelector(d, channel=c))
        
        return cls(*selectors)

    def check_well_defined(self):
        """
        Check if the :code:`AxisSelector` s are configured in a self-consistent
        manner.

        In particular, this method checks if there are :code:`DimNameSelector` s with 
        duplicate names or :code:`AxisSelector` s with duplicate indices.

        :raises: :code:`ValueError` if the check fails

        """
        flag = False
        names = [sel.dimname for sel in self.selectors if isinstance(sel, DimNameSelector)]

        if len(names) != len(set(names)):
            flag = True

        axes = [sel.axis for sel in self.selectors if isinstance(sel, IndexSelector)]
        if len(axes) != len(set(axes)):
            flag = True

        if flag:
            raise ValueError('Axis selectors not consistent')

    def cast_and_update(self, X: XTensor, channel: str|None=None) -> XTensor:
        """
        Cast the input tensor according to the template and update self's
        internal state (i.e. dimension names and coordinates).

        """
        axesp = [sel.select_axis(X, channel) for sel in self.selectors]
        assert permutation_well_defined(axesp, X.rank), 'Casting impossible'

        X1 = permute(X, axesp)

        self._dims = mergedims(self._dims, X1)
        self._coords = mergecoords(self._coords, X1)

        return X1

    def clear(self):
        """
        Reset the internal states.

        """
        self._dims = [None for _ in range(len(self.selectors))]
        self._coords= [None for _ in range(len(self.selectors))]

    @property
    def dims(self) -> Dims:
        """
        The dimension names gathered from casting tensors.
        """
        return self._dims

    @property
    def coords(self) -> Coords:
        """
        The coordinates names gathered from casting tensors.
        """

        return self._coords

    def __repr__(self) -> str:
        s = 'Dimension Template'
        s1 = 'dim.    '
        s2 = 'channel '

        for selector in self.selectors:
            label = selector.label
            channel = str(selector.channel)

            width = max(len(label), len(channel)) + 1

            s1 += f'{label:{width}}'
            s2 += f'{channel:{width}}'


        return s + '\n' + s1 + '\n' + s2

