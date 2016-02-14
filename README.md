# binary_net by chainer

This is an experimental code for reproducing [1]'s result using chainer. 
No optimization is used for binary operations. I just binalize weight and activation at computation, and use a straight through estimator for gradient computation. 

- [1] "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1", Matthieu Courbariaux, Yoshua Bengio
http://arxiv.org/abs/1602.02830

Code is almost equivalent to chainer/examples/mnist/ except:

- Use binary weight, binary activation, batch_normalization (net.py, bst.py, link_binary_linear.py function_binary_linear.py)
- Use weight clip, optimizer.add_hook(weight_clip.WeightClip()) (weight_clip.py)


Usage
```
# cpu
./train_mnist.py

# gpu (use device id=0)
./train_mnist.py --gpu=0
```

Result
---
load MNIST dataset
epoch 1
graph generated
train mean loss=0.573861178756, accuracy=0.92756666926
test  mean loss=0.473955234885, accuracy=0.957400003672
epoch 2
train mean loss=0.456328810602, accuracy=0.963833337426
test  mean loss=0.436628208458, accuracy=0.966100006104
epoch 3
train mean loss=0.431186137001, accuracy=0.970866675178
test  mean loss=0.425710965991, accuracy=0.968000004292
epoch 4
train mean loss=0.417045980394, accuracy=0.975233343144
test  mean loss=0.417223671675, accuracy=0.969800002575
epoch 5
train mean loss=0.409991853635, accuracy=0.977583343883
test  mean loss=0.407217691839, accuracy=0.972200006247
epoch 6
train mean loss=0.400645414094, accuracy=0.979883343577
test  mean loss=0.40729173243, accuracy=0.972400006652
epoch 7
train mean loss=0.395223465959, accuracy=0.981483343343
test  mean loss=0.402929984331, accuracy=0.972300007343
epoch 8
train mean loss=0.389928704053, accuracy=0.983366676569
test  mean loss=0.402315998375, accuracy=0.97280000627
epoch 9
train mean loss=0.389456737339, accuracy=0.983600010673
test  mean loss=0.39955814153, accuracy=0.973400005698
epoch 10
train mean loss=0.385094682376, accuracy=0.984783343176
test  mean loss=0.401046113968, accuracy=0.972200005651
epoch 11
train mean loss=0.38257016028, accuracy=0.986000010371
test  mean loss=0.393966214061, accuracy=0.974400005937
epoch 12
train mean loss=0.379689370046, accuracy=0.986583343049
test  mean loss=0.396037294269, accuracy=0.974900006056
epoch 13
train mean loss=0.378962427129, accuracy=0.986783343355
test  mean loss=0.392184624076, accuracy=0.974600006342
epoch 14
train mean loss=0.375957165956, accuracy=0.987533342044
test  mean loss=0.394931056798, accuracy=0.974400007725
epoch 15
train mean loss=0.375070895106, accuracy=0.988500009278
test  mean loss=0.393464969695, accuracy=0.974700006843
epoch 16
train mean loss=0.37365236491, accuracy=0.988550009727
test  mean loss=0.397632206976, accuracy=0.972800005078
epoch 17
train mean loss=0.372001686394, accuracy=0.989216675361
test  mean loss=0.392721504271, accuracy=0.973500006795
epoch 18
train mean loss=0.369835597078, accuracy=0.989616675278
test  mean loss=0.388620298207, accuracy=0.975900007486
epoch 19
train mean loss=0.369737090866, accuracy=0.98986667484
test  mean loss=0.391184872091, accuracy=0.974400007725
epoch 20
train mean loss=0.368466852009, accuracy=0.98983334144
test  mean loss=0.389251522124, accuracy=0.976600005627
save the model
save the optimizer
```
