# HW1
## Problem Description:
Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set.

## Implementation:
I implemented a single layer neural network with mini batch SGD to train my
model. Sigmoid function is used as hyper param e ter function and softmax function is
used as activation function of output layer. For calculating gradient of cross entropy
error, I used forward and backward function in the lecture slides. The whole frame is
as follows:
1. Importing data and separating them as training data, training label, testing
data and t esting label.
2. Initializing model parameter θ W1, B1, C, B2 via rand om numbers
generated from normal distribution and divided by corresponding square root
of neuron numbers.
3. Implementing forward and backward function and storing repeated variables
which are X and Z to accelerate training.
4. Setting initial learning rate 0.5 )), epoch 40 and batch size 20
5. When training, for each epoch, I shuffled data and used piece wise schedule
to avoid overshooting of gradient decent. For each batch of data, I used back
propagation to calculate sum of gradient and took the average of it.
6. After training, using test data to test the accuracy of my training model.

## Result:
When using hidden layer s as 50, epoch as 40, batch size as 20 and learning as 0.5
in the first 25 epochs and 0.4 as the remaining 15 e pochs. Accuracy is above 98% for
training data and above 97% for testing data.
