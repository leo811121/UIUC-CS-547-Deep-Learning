# HW3
## Problem Description
Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout,
(B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy 
(i) using the heuristic prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90% Test Accuracy.
## Implementation
Implementation:
I implemented a multi channel CNN with multiple channel and Adam Adam
optimization to train my model. In the CNN, some layer I follow the frame from the
lecture slides to implemented dropout and batch normalization to avoid overfitting
and gradient vanishing problems. Before training, I did some data augmentation to
preprocess training and testing data. When testing data, I implemented normal method
model.train( train())), he u ris tic method (model.eval()) and Monte Carlo method on testing
data.
The whole frame is as follows:
1. For importing data, I import ed data and separat ed them as training data,
training label, testing data and testing label and t hen shuffle data . In the end I
set training data batch size as 128 and testing data size as 100
2. For data augmentation, I implement ed crop and horizontal flip
3. For building model, I completely f ollow the lecture to build my CNN model.
However, in each layer, I added ReLU function as activation fu n ction. For
layer existing dropout, I set dropout s possibility as 0.05.
4. For training the model, I us ed Adam optimiz ation and set learning as 0.01,
epoch as 40. But if train data s accuracy are over 88%, the program will
implemented model testing data. If testing data accuracy is over 80%, the
program will stop.
5. For testing data, I implemented normal method (model.train()), heuristic
method(model.eval()) and Monte Carlo method on testing For Monte
Carlo method, I set the for loop as 100 t imes and then calculated the average
of all result.
## Result:
When using filter size as 3 X3 filter number as 3, epoch as 3 , and learning as
0.01. Accuracy is above 94% for training data and above 95 % for testing


