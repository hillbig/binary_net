import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class BinaryStraightThrough(function.Function):

    """Exponential Linear Unit."""

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = numpy.where(x>=0, 1, -1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x >= 0 ? 1 : -1', 'binary_straight_through_fwd')(
                x[0])
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = x[0] > 1 or x < -1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = (x >= 1 || x <= -1) ? 0 : gy', 'bainry_straight_through_bwd')(
                x[0], gy[0])
        return gx,


def binary_straight_through(x):
    """Exponential Linear Unit function.

    This function is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.

    where :math:`\\alpha` is a parameter.
    See: http://arxiv.org/abs/1511.07289

    Args:
        x (~chainer.Variable): Input variable.
        alpha (float): Parameter :math:`\\alpha`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return BinaryStraightThrough()(x)
