import numpy as np
from .. import numpy as xtnp
from .. import tensor as xtt


def confusion_matrix(truth: np.ndarray, pred: np.ndarray, /, *, n_classes: int) -> np.ndarray:
    '''
        X: (*, M)
        Y: (*, M)
    '''
    z = truth* n_classes + pred
    cmat = xtnp.bincount(z, N=n_classes**2)
    return cmat.reshape(*cmat.shape[:-1], n_classes, n_classes)


def get_confmat_function(target_dim: str, truth_dim: str, pred_dim: str, n_classes: int):
    @xtt.generalize_at_0
    @xtt.generalize_at_1
    def wrapped_confmat(truth: xtt.XTensor, pred: xtt.XTensor) -> xtt.XTensor:
        
        X = xtt.name_dim_if_absent(truth, -1, target_dim)
        Y = xtt.name_dim_if_absent(pred, -1, target_dim)

        X = xtt.dimslast(truth, [target_dim])
        Y = xtt.dimslast(pred, [target_dim])

        x, y, dims, coords = xtt.vanilla_broadcaster(X, Y)
        cm = confusion_matrix(x, y, n_classes=n_classes)

        C = xtt.XTensor(cm,
                dims=dims[:-1] + [truth_dim, pred_dim],
                coords=coords[:-1] + [None, None]
            )
        return C

    return wrapped_confmat

