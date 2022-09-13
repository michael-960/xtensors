from collections.abc import Callable
from typing_extensions import ParamSpec
import numpy as np

from ..tensor import promote_binary_operator, XTensor, generalize_2
from ..tensor.typing import Function_2Args, Dims, Coords


from ..tensor.basic_utils import dimslast, dimsfirst, flatten

from .. import numpy as xtnp


def _confmat(X: np.ndarray, Y: np.ndarray, /, *, n_classes: int) -> np.ndarray:
     _z = X * n_classes + Y
     cmat = xtnp.bincount(_z, N=n_classes**2)
     return cmat.reshape(*cmat.shape[:-1], n_classes, n_classes)


def confusion_matrix(
        target_dim: str, class_truth_dim: str, class_pred_dim: str, n_classes: int
        ): 

    def convert(dims: Dims, coords: Coords):
        return dims[:-1] + [class_truth_dim, class_pred_dim], coords[:-1] + [None, None]

    @with_broadcast(dimcoord_converter=convert)
    def __confmat(X: np.ndarray, Y: np.ndarray):
        return _confmat(X, Y, n_classes=n_classes)

    @generalize_2
    def confmat(X: XTensor, Y: XTensor) -> XTensor:
        X1 = dimslast(X, [target_dim])
        Y1 = dimslast(Y, [target_dim])
        _cm = __confmat(X1, Y1)
        
        _cm.set_dim(-2, class_truth_dim)
        _cm.set_dim(-1, class_pred_dim)
        return _cm

    return confmat
