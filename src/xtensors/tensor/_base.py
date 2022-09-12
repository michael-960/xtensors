from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar
from typing_extensions import TypeVarTuple, Unpack
from numpy.typing import ArrayLike, NDArray
import numpy as np



class XTensor:
    def __init__(self, data: NDArray,
        dims: Optional[Sequence[str|None]]=None,
        coords: Optional[Sequence[Sequence[Any]|None]]=None,
            ) -> None:

        self.data = data

        self._dim_axis_dict: Dict[str,int] = dict()

        self._dims: List[str|None]
        self._coords: List[Sequence[Any]|None]

        self.set_dims(dims)
        self.set_coords(coords)

    def viewcopy(self) -> XTensor:
        return XTensor(self.data, self.dims, self.coords)
    

    @property
    def dims(self) -> Tuple[str|None,...]:
        return tuple(self._dims)

    @property
    def coords(self) -> Tuple[Sequence[Any]|None,...]:
        return tuple(self._coords)

    @property
    def shape(self) -> Tuple[int,...]:
        return self.data.shape

    @property
    def rank(self) -> int:
        return len(self.data.shape)

    def get_axis(self, dim: str) -> int:
        return self._dim_axis_dict[dim]
            
    def set_dims(self, dims: Sequence[str|None]|None) -> None:
        if dims is None:
            self._dims = [None for _ in self.data.shape]
        else:
            assert len(dims) == len(self.data.shape)
            _dims_without_none = [dim for dim in dims if dim is not None]
            assert len(_dims_without_none) == len(set(_dims_without_none))
            self._dims = list(dims)

            for axis, dim in enumerate(dims):
                if dim is not None:
                    self._dim_axis_dict[dim] = axis

    def set_dim(self, axis: int, dim: str|None) -> None:
        if axis < 0: axis = self.rank + axis
        new_dims = self._dims.copy()
        new_dims[axis] = dim
        self.set_dims(new_dims)

    def set_coords(self, coords: Sequence[Sequence[Any]|None]|None) -> None:
        if coords is None:
            self._coords = [None for _ in self.data.shape]
        else:
            assert len(coords) == len(self.data.shape)
            for axis, coord in enumerate(coords):
                if coord is not None:
                    assert len(coord) == self.data.shape[axis]

            self._coords = list(coords)

    def set_coord(self, axis: int, coord: Sequence[Any]|None) -> None:
        if axis < 0: axis = self.rank + axis
        new_coords = self._coords.copy()
        new_coords[axis] = coord
        self.set_coords(new_coords)

    
    def __repr__(self):
        _repr = 'Tensor\n'
        _repr += f'shape={self.shape}\n'
        _repr += f'dims={self.dims}\n'

        _repr += f'coords:\n'
        for axis, coord in enumerate(self.coords):
            if coord is None:
                _repr += f'{axis}: None\n'
            if coord is not None:
                _repr += f'{axis}: {coord[0]}..{coord[-1]}\n'


        _repr += self.data.__repr__()

        return _repr

    def __neg__(self) -> XTensor:
        return XTensor(data=-self.data, dims=self._dims, coords=self._coords)


