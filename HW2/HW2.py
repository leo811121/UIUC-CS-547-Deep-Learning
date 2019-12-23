#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import numpy.linalg as la
import h5py
import time
import copy
from random import randint

#Implementation of stochastic gradient descent algorithm
def sigmoid_function(z):
    return np.maximum(z,0)

def sigmoid_function_der(z):  #need to modify
    z[z<=0] = 0
    z[z>0] = 1
    return z

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x,y, model):
    global Z
    global U
    global H
    x = np.reshape(x,(num_inputs,num_inputs))
    Z = np.zeros((num_inputs-k_y+1,num_inputs-k_x+1,C))
    for c in range(C):
        for i in range(num_inputs-k_y+1):
            for j in range(num_inputs-k_x+1):
                Z[i,j,c] = np.sum(model['K'][:,:,c]*x[i:i+k_y,j:j+k_x])
    H = sigmoid_function(Z)
    U = np.zeros((num_outputs,1))
    for output in range(num_outputs):
        U[output] = np.sum(model['W'][output,:,:,:]*H) + model['b'][output]
    f_x_theta = softmax_function(U)   
    return f_x_theta

def backward(x,y,p, model, model_grads):
    x = np.reshape(x,(num_inputs,num_inputs))
    output=np.zeros((num_outputs,1))
    output[y]=1
    model_grads['b'] = -(output-p) #p or U need to be comfirmed
    
    delta = np.zeros((num_inputs-k_y+1, num_inputs-k_x+1,C))
    
    for c in range(C):
        for i in range(num_inputs-k_y+1):
            for j in range(num_inputs-k_x+1):
                #delta[i,j,c] = np.sum(model_grads['b']*model['W'][:,i,j,c])
                delta[i,j,c] = np.sum(np.reshape(model_grads['b'],(1,10))*model['W'][:,i,j,c])
               
    '''   
    temp = np.zeros((num_inputs-k_y+1, num_inputs-k_x+1,C))
    for c in range(C):
        temp[:,:,c] = sigmoid_function_der(Z)[:,:,c]*delta[:,:,c]
    '''
    model_grads['K'] = np.zeros((k_y, k_x, C))
    for c in range(c):
        for i in range(k_y):
            for j in range(k_x):
                model_grads['K'][i,j,c] = np.sum(x[i:i+num_inputs-k_y+1, j:j+num_inputs-k_x+1]*(sigmoid_function_der(Z)[:,:,c]*delta[:,:,c]))  
                 
    for k in range(num_outputs):
        model_grads['W'][k,:,:,:] = model_grads['b'][k]*H
    #print(model_grads['K'].shape)
    return model_grads

def test(x,y):
    print('test data')
    total_correct = 0
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
    print('test',total_correct/np.float(len(x_test) ) )
    return total_correct/np.float(len(x_test) ) 

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

#number of inputs
num_inputs = 28 #width of a picture
#number of outputs
num_outputs = 10
#filter size
k_y = 3
k_x = 3
#number of filter
C = 3

#building model's initial value
#why multiply 0.1 & 0.001?
model = {}
model['K'] = np.random.randn(k_y, k_x, C)/((num_inputs*num_inputs)**0.5)
model['W'] = np.random.randn(num_outputs, num_inputs-k_y+1, num_inputs-k_x+1,C)/((num_inputs*num_inputs)**0.5)
model['b'] = np.random.randn(num_outputs,1)
model['b'] = np.reshape(model['b'],(-1,1))/((num_inputs*num_inputs)**0.5)
model_grads = copy.deepcopy(model)

#a = forward(x_train[1],y_train, model)
#b = backward(x_train[1],y_train, a, model, model_grads)

#number of epochs
num_epochs = 3
#import time
#time1 = time.time()
LR = 0.01
for epochs in range(num_epochs):
    total_correct = 0
    for i in range(len(y_train)):
        '''
        if (i>3000):
            LR = 0.005
        '''
        index_Rand = randint(0,len(y_train)-1)
        print('epoch',epochs,'i',i)
        '''
        x = x_train[i]
        y = y_train[i]
        '''
        x = x_train[index_Rand,:]
        y = y_train[index_Rand]
        p = forward(x, y, model)
                                      
        prediction = np.argmax(p)                              
        if (prediction == y):
            total_correct += 1
        correct_record = total_correct/np.float(i+1)
        print('correctness',correct_record ) 
        
        model_grads = backward(x,y,p, model, model_grads)
        
        model['K'] = model['K'] - LR * model_grads['K']
        model['W'] = model['W'] - LR * model_grads['W']
        model['b'] = model['b'] - LR * model_grads['b']
        
        '''
        if (correct_record > 0.96):
            test_result = test(x_test,y_test)
            if (test_result > 0.94): 
                break
        '''
#time2 = time.time()
#print(time2-time1)
######################################################

#test data
print('test data')
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




