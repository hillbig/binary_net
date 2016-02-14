# binary_net

This is an experimental code for reproducing [1] result. 
No optimization is used for binary operations. I just binalize weight and activation and use straight through estimator for gradient computation. 

- [1] "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1", Matthieu Courbariaux, Yoshua Bengio
http://arxiv.org/abs/1602.02830

# use cpu
python train_mnist.py

# use gpu
train_mnist.py --gpu=0
