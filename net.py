import chainer
import chainer.functions as F
import chainer.links as L
import link_binary_linear
import binary_straight_through

class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=link_binary_linear.BinaryLinear(n_in, n_units),
            l2=link_binary_linear.BinaryLinear(n_units, n_units),
            l3=link_binary_linear.BinaryLinear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = binary_straight_through.binary_straight_through(self.l1(x))
        h2 = binary_straight_through.binary_straight_through(self.l2(h1))
        return self.l3(h2)


