Argument Functions and Coordinate Functions
============================================

Argument functions, when fed with a single :py:class:`xtensors.TensorLike` object, return
an :py:class:`XTensor` instance with integer data representing the **indices**
of the original tensor. An example is :py:func:`xtensors.argmax`, which returns
the indices where the maximum values occur.

Similarly, coordinate functions return the **coordinates** associated with the
axes of the original tensor.

.. autofunction:: xtensors.argmax


