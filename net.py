import chainer
import chainer.functions as F
import chainer.links as L
import link_binary_linear
import bst

class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=link_binary_linear.BinaryLinear(n_in, n_units),
            b1=L.BatchNormalization(n_units),
            l2=link_binary_linear.BinaryLinear(n_units, n_units),
            b2=L.BatchNormalization(n_units),
            l3=link_binary_linear.BinaryLinear(n_units, n_out),
            b3=L.BatchNormalization(n_out),
        )
        self.train = True

    def __call__(self, x):
        h1 = bst.bst(self.b1(self.l1(x), test=not self.train))
        h2 = bst.bst(self.b2(self.l2(h1), test=not self.train))
        return self.b3(self.l3(h2), test=not self.train)


