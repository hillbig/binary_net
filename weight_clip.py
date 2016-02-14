from chainer import cuda

class WeightClip(object):

    """Optimizer hook function for weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'WeightClip'

    def __init__(self):
        pass

    def __call__(self, opt):
        if cuda.available:
            kernel = cuda.elementwise(
                '', 'T p', 'p = (p < -1) ? -1 : (p > 1) ? 1 : p', 'weight_clip')

        for param in opt.target.params():
            p = param.data
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    p = numpy.min(1, numpy.max(-1, p))
                else:
                    kernel(p)
