#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numpy.linalg as la
import h5py
import time
import copy
from random import randint


#Implementation of stochastic gradient descent algorithm
def sigmoid_function(z):
    return 1.0/(1.0 + np.exp(-z))

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x,y, model):
    #Store X and Z
    global x_mod
    global Z
    x_mod = np.reshape(x,(-1,1))
    Z = model['W1']@ x_mod + model['b1']
    H_i = sigmoid_function(Z)
    U = model['C']@ H_i + model['b2']
    f_x_theta = softmax_function(U)
    return f_x_theta

def backward(x,y,p, model, model_grads):
    #x_mod = np.reshape(x,(-1,1))
    #Z = np.dot(model['W1'], x_mod) + model['b1']
    H_i = sigmoid_function(Z)
    output=np.zeros((num_outputs,1))
    output[y]=1
    model_grads['b2'] = -(output-p) 
    model_grads['C'] = np.dot(model_grads['b2'],H_i.T)
    
    delta = np.dot(model['C'].T,model_grads['b2'])
    sigmoid_der = H_i*(1-H_i)
    
    model_grads['b1'] = delta * sigmoid_der
    model_grads['W1'] = np.dot(model_grads['b1'],x_mod.T)
    return model_grads

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
#number of hidden layers
num_h = 50

#building model's initial value
model = {}
model['W1'] = np.random.randn(num_h,num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(num_h,1) / np.sqrt(num_h)
model['b1'] = np.reshape(model['b1'],(-1,1))
model['C'] = np.random.randn(num_outputs,num_h) / np.sqrt(num_h)
model['b2'] = np.random.randn(num_outputs,1) / np.sqrt(num_outputs)
model['b2'] = np.reshape(model['b2'],(-1,1))
model_grads = copy.deepcopy(model)

#number of batchs
batch = 20
#number of epochs
num_epochs = 40


import time
time1 = time.time()
LR = 0.05
for epochs in range(num_epochs):
    index_Permute = np.random.permutation(len(y_train))
    x_train = x_train[index_Permute,:]
    y_train = y_train[index_Permute]
    print('epoch',epochs)
    total_correct = 0
    
    #Learning rate schedule
    if (epochs > 25):
        LR = 0.04
        
    for i in range(0, len(y_train), batch):
        x_train_batch = x_train[i:i+batch,:] 
        y_train_batch = y_train[i:i+batch]
        new_model_grads = {}
        new_model_grads['W1'] = 0 
        new_model_grads['b1'] = 0 
        new_model_grads['C']  = 0
        new_model_grads['b2'] = 0
        
        for j in range(batch):
            x = x_train_batch[j]
            y = y_train_batch[j]
            p = forward(x, y, model)
            prediction = np.argmax(p)
            if (prediction == y):
                total_correct += 1
            model_grads = backward(x,y,p, model, model_grads)
            new_model_grads['W1'] += model_grads['W1']
            new_model_grads['b1'] += model_grads['b1'] 
            new_model_grads['C'] += model_grads['C'] 
            new_model_grads['b2'] += model_grads['b2']

        model['W1'] = model['W1'] - LR/batch * new_model_grads['W1']
        model['b1'] = model['b1'] - LR/batch * new_model_grads['b1']
        model['C'] = model['C'] - LR/batch * new_model_grads['C']
        model['b2'] = model['b2'] - LR/batch * new_model_grads['b2']
    print('correctness',total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('test',total_correct/np.float(len(x_test) ) )


# In[ ]:





# In[ ]:





# In[ ]:




