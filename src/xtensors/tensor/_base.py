from __future__ import annotations

import numpy as np

from .broadcast import vanilla_broadcaster

from ._decors import promote_binary_operator

from .basic_utils._base import to_xtensor
from .basic_utils._misc import strip

from ._slice import TensorIndexer

from typing import TYPE_CHECKING

from .typing import DimLike, DimsLike, TensorLike, Array

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
    from .broadcast import Broadcaster
    from .typing import BinaryOperator

def inject_broadcast(broadcaster: Broadcaster):
    promote = promote_binary_operator(broadcaster)
    def wrapper(f: Callable[[XTensor], BinaryOperator[np.ndarray]]
        ):
        def wrapped(self: XTensor, other: TensorLike, /) -> XTensor:
            binop = promote(f(self))
            if isinstance(other, XTensor):
                return binop(self, other)
            try:
                return binop(self, to_xtensor(other))
            except TypeError:
                return NotImplemented
        return wrapped
    return wrapper


class XTensor:
    r"""
    A wrapper class of :code:`np.ndarray` that attaches names to each axis.
    "axis" and "dimension" are synonyms in this context.

    """
    def __init__(self, data: Array,
        dims: Optional[Sequence[str|None]]=None,
        coords: Optional[Sequence[Sequence[Any]|NDArray[Any]|None]]=None,
    ) -> None:
        """
        :param data: the underlying array object, accepts any object that implements the :code:`__array__` protocol
        :param dims: a sequence of strings specifying the names of each axis
        :param coords: a sequence of 1D arrays specifying the coordinates of each axis

        By default, each dimension has :code:`None` as both its name and coordinates.

        """

        self.data = data.__array__()

        self._dim_axis_dict: Dict[str,int] = dict()

        self._dims: List[str|None]
        self._coords: List[NDArray[Any]|None]

        self.set_dims(dims)
        self.set_coords(coords)

    def viewcopy(self) -> XTensor:
        r"""
        :return: a new :code:`XTensor` object with the *same* underlying :code:`data`. 
                Useful when one wishes to attach different metadata to the same array.

        """
        return XTensor(self.data, self.dims, self.coords)
    
    def item(self) -> float:
        r"""
        :return: :code:`data.item()`

        :raises: :code:`ValueError` if :code:`data` is not scalar

        """
        return self.data.item()

    @property
    def dims(self) -> Tuple[str|None,...]:
        """
        A tuple of strings or None corresponding to each axis's name.

        """
        return tuple(self._dims)

    @property
    def coords(self) -> Tuple[NDArray[Any]|None,...]:
        r"""
        A tuple of :code:`np.ndarray` or None corresponding to each axis's
        coordinates.

        """

        return tuple(self._coords)

    @property
    def shape(self) -> Tuple[int,...]:
        r"""
        A tuple of :code:`int`, same as :code:`data.shape`

        """
        return self.data.shape

    @property
    def rank(self) -> int:
        r"""
        Number of axes, same as :code:`data.ndim`
        """
        return len(self.data.shape)

    def get_axis(self, dim: DimLike) -> int:
        r"""
        Performs an axis lookup with a :code:`DimLike` object

        :return: an integer representing the resulting axis

        :raises: TypeError if :code:`dim` is not a valid :code:`DimLike` object

        """
        try:
            dim = dim.__get_dimname__() # type: ignore
        except AttributeError: pass

        if isinstance(dim, str): return self._dim_axis_dict[dim]
        if isinstance(dim, int):
            if dim < 0: dim = dim + self.rank
            if dim < 0 or dim >= self.rank: raise ValueError(f'Invalid axis: {dim}, rank={self.rank}')
            return dim
        
        try:
            assert isinstance(dim, tuple) 
            assert isinstance(dim[0], str)
            assert isinstance(dim[1], int)
            assert len(dim) == 2
        except AssertionError as e:
            raise TypeError(f'Dimlike should be int, str, or (str, int), but received {dim} instead') from e


        try:
            return self.get_axis(dim[0])
        except KeyError:
            return self.get_axis(dim[1])

    def get_axes(self, dims: DimLike|DimsLike|None) -> List[int]:
        r"""
        Performs multiple axis lookups

        :return: A list of axes (integers)

        """
        if dims is None:
            return [i for i in range(self.rank)]
        try:
            dims = dims.__get_dimname__() # type: ignore
        except AttributeError: pass

        if isinstance(dims, (int, str, tuple)): return [self.get_axis(dims)]

        if not isinstance(dims, list):
            raise ValueError(f'Invalid dimension: {dims}')

        axes = [self.get_axis(dim) for dim in dims]

        if len(axes) != len(set(axes)):
            raise ValueError(f'Duplicate axes {axes}')

        return axes
            
    def set_dims(self, dims: Sequence[str|None]|None) -> None:
        r"""
        Set axis names.

        """
        if dims is None:
            self._dims = [None for _ in self.data.shape]
        else:
            if len(dims) != len(self.data.shape):
                raise ValueError(f'Invalid dimension names {dims} for tensor with shape {self.data.shape}')

            if '' in dims:
                raise ValueError(f'Invalid dimension names {dims}')

            _dims_without_none = [dim for dim in dims if dim is not None]
            if len(_dims_without_none) != len(set(_dims_without_none)):
                raise ValueError(f'Duplicate dimension names in {dims}')

            self._dims = list(dims)

            for axis, dim in enumerate(dims):
                if dim is not None:
                    self._dim_axis_dict[dim] = axis

    def set_dim(self, dim: DimLike, newdim: str|None) -> None:
        r"""
        Set the name of a single axis.

        """
        axis = self.get_axis(dim)
        new_dims = self._dims.copy()
        new_dims[axis] = newdim
        self.set_dims(new_dims)

    def get_coord(self, dim: DimLike) -> NDArray|None:
        r"""
        Return the coordinates of the given axis.

        :return: If the axis has coordinates, an :code:`np.ndarray` is returned, else :code:`None` is returned.

        """
        axis = self.get_axis(dim)
        return self._coords[axis]

    def set_coords(self, coords: Sequence[Sequence[Any]|NDArray[Any]|None]|None) -> None:
        r"""
        Set coordinates.

        """

        if coords is None:
            self._coords = [None for _ in self.data.shape]
        else:
            coords_clean: List[NDArray[Any]|None] = []
            if len(coords) != len(self.data.shape):
                raise ValueError(f'Received {len(coords)} coordinates for tensor with shape {self.data.shape}')

            for axis, coord in enumerate(coords):
                if coord is not None:
                    if len(coord) != self.data.shape[axis]:
                        raise ValueError(
                                f'Received coordinates with {len(coord)} elements at axis {axis} '+
                                f'for tensor with shape {self.data.shape}')

                    coords_clean.append(np.array(coords[axis]))
                else:
                    coords_clean.append(None)
            self._coords = coords_clean

    def set_coord(self, dim: DimLike, coord: Sequence[Any]|NDArray[Any]|None) -> None:
        r"""
        Set the coordinates of a single axis.

        """

        axis = self.get_axis(dim)
        new_coords = self._coords.copy()
        if coord is None:
            new_coords[axis] = None
        else:
            new_coords[axis] = np.array(coord)
        self.set_coords(new_coords)


    def slc(self, dim: DimLike, slc: slice) -> XTensor:
        """
        :return: A new XTensor object with :code:`dim` sliced by :code:`slice`.

        """
        axis = self.get_axis(dim)

        slices: List[slice] = [slice(None, None, None)]*self.rank
        slices[axis] = slc

        data = self.data[tuple(slices)]
        coords = list(self.coords)
        
        coord = coords[axis]
        if coord is not None:
            coords[axis] = coord[slc]

        return XTensor(data, dims=self.dims, coords=coords)


    def get(self, dim: DimLike, index: int) -> XTensor:
        r"""
        :return: A new XTensor object with :code:`dim` indexed by :code:`index`.

        """
        axis = self.get_axis(dim)
        
        slices: List[slice|int] = [slice(None, None, None)]*self.rank
        slices[axis] = index

        data = self.data[tuple(slices)]

        dims = strip(self.dims, [axis])
        coords = strip(self.coords, [axis])
        
        return XTensor(data, dims=dims, coords=coords)

    def __getitem__(self, slices: TensorIndexer|Tuple[TensorIndexer,...]) -> XTensor:
        """

        """
        if isinstance(slices, tuple):
            _Y = slices[0].index(self)
            if len(slices) == 1: 
                return _Y
            return _Y.__getitem__(tuple(slices[1:]))
        return slices.index(self)

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

    @inject_broadcast(vanilla_broadcaster)
    def __add__(self): return lambda X, Y: X+Y

    @inject_broadcast(vanilla_broadcaster)
    def __radd__(self): return lambda X, Y: X+Y

    @inject_broadcast(vanilla_broadcaster)
    def __sub__(self): return lambda X, Y: X-Y

    @inject_broadcast(vanilla_broadcaster)
    def __rsub__(self): return lambda X, Y: Y-X

    @inject_broadcast(vanilla_broadcaster)
    def __mul__(self): return lambda X, Y: X*Y

    @inject_broadcast(vanilla_broadcaster)
    def __rmul__(self): return lambda X, Y: X*Y

    @inject_broadcast(vanilla_broadcaster)
    def __truediv__(self): return lambda X, Y: X/Y

    @inject_broadcast(vanilla_broadcaster)
    def __rtruediv__(self): return lambda X, Y: Y/X

    @inject_broadcast(vanilla_broadcaster)
    def __pow__(self): return lambda X, Y: X**Y

    @inject_broadcast(vanilla_broadcaster)
    def __rpow__(self): return lambda X, Y: Y**X

    @inject_broadcast(vanilla_broadcaster)
    def __eq__(self): return lambda X, Y: X==Y

    @inject_broadcast(vanilla_broadcaster)
    def __lt__(self): return lambda X, Y: X<Y

    @inject_broadcast(vanilla_broadcaster)
    def __gt__(self): return lambda X, Y: X>Y

    @inject_broadcast(vanilla_broadcaster)
    def __le__(self): return lambda X, Y: X<=Y

    @inject_broadcast(vanilla_broadcaster)
    def __ge__(self): return lambda X, Y: X>=Y

    def __array__(self): return self.data



