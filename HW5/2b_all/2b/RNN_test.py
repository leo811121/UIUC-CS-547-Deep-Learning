import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io
from RNN_model import RNN_model


glove_embeddings = np.load('/u/training/tra335/HW5/1a/preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
x_test = []
with io.open('/u/training/tra335/HW5/1a/preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = torch.load('rnn.model')
model.cuda()

batch_size = 200
no_of_epochs = 500
#L_Y_train = len(y_train)
L_Y_test = len(y_test)


train_loss = []
train_accu = []
test_accu = []


for epoch in range(no_of_epochs):

    # ## test
    if((epoch+1)%1)==0:
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
    
        I_permutation = np.random.permutation(L_Y_test)
        for i in range(0, L_Y_test, batch_size):

###################################################################################################
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = (epoch + 1) * 50
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            x_input = glove_embeddings[x_input]
            y_input = y_test[I_permutation[i:i+batch_size]]

            data = Variable(torch.FloatTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()
##################################################################################################
            with torch.no_grad():
                loss, pred = model(data,target)

            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("Seqence:", sequence_length," Accuracy: %.2f" % (epoch_acc*100.0), "loss: %.4f" % epoch_loss)
       # print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)
