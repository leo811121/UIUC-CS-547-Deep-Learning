# HW2
## Problem Description
Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). You should write your own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve at least 94% accuracy on the Test Set.
## Implementation
I implemented a single layer CNN with multiple channel and SGD to train my
model. ReLU function is used as activation function and S oft max function is used as
activation function of output layer . For calculating gradient of cross entropy error, I
used forward and backward function in the lecture slides. Since delta in the backwar d
propagation needs the derivative of ReLU function.
The whole frame is as follows:
1. Importing data and separating them as training data, train ing label, testing
data and testing label.
2. Setting filter size, I choose the size as 3X3. And the number of filter I choose
is 3.
3. Initializing model parameter θ K, W, b ) via Xavier/He initialization
4. Implementing forward and backward function and storin g repeated variables
which are Z , U, H to accelerate training.
5. S etting initial learning rate (0.01 and epoch (3
6. When training, for each epoch, I randomize training data. Fo r each batch of
data, I used back propagation to calculate gradient
7. After training, using test data to test the accuracy of my training model.
## Result:
![image](https://ibb.co/DMXc0q2)
