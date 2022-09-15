from __future__ import annotations


from .. import tensor as xtt


class Functional:
    def __init__(self):
        self.name: str = 'NOTIMPLEMENTED_FUNCTIONAL'

    def __call__(self, x: xtt.TensorLike) -> xtt.XTensor:
        raise NotImplementedError


class Identity(Functional):
    def __init__(self, name='I'):
        self.name = name

    def __call__(self, x: xtt.TensorLike) -> xtt.XTensor:
        return xtt.to_xtensor(x)


class Pipe(Functional):
    '''
        Pipe(f1, f2, ..., fn)(x) = fn(...f2(f1(x))...)
    '''
    def __init__(self, *f: Functional, delim: str='.'):
        self.f = f
        self.delim = delim
        self.name = delim.join([_f.name for _f in self.f])

    def __call__(self, x: xtt.TensorLike) -> xtt.XTensor:
        _y = xtt.to_xtensor(x)
        for _f in self.f:
            _y = _f(_y)
        return _y
