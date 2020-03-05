# 1a: Bag of Words Without GloVe Features
A bag of words model is one of the most basic models for document classification. It is a way of handling varying length inputs (sequence of token ids) to get a single output (positive or negative sentiment).

The token IDs can be thought of as an alternative representation of a 1-hot vector. A word embedding layer is usually a d by V matrix where V is the size of the vocabulary and d is the embedding size. Multiplying a 1-hot vector (of length V) by this matrix results in a d dimensional embedding for a particular token. We can avoid this matrix multiplication by simply taking the column of the word embedding matrix corresponding with that particular token.

A word embedding matrix doesn’t tell you how to handle the sequence information. In this case, you can picture the bag of words as simply taking the mean of all of the 1-hot vectors for the entire sequence and then multiplying this by the word embedding matrix to get a document embedding. Alternatively, you can take the mean of all of the word embeddings for each token and get the same document embedding. Now we are left with a single d dimensional input representing the entire sequence.

Note that the bag of words method utilizes all of the tokens in the sequence but loses all of the information about when the tokens appear within the sequence. Obviously the order of words carries a significant portion of the information contained within written text but this technique still works out rather well.

Create a new directory ‘1a/’ within your assignment directory. You will create two python files, BOW_model.py and BOW_sentiment_analysis.py. The first is where we will define our bag of words pytorch model and the second will contain the train/test loops.

BOW_model.py
```
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()
        ## will need to define model architecture   
        
    def forward(self, x, t):
        # will need to define forward function for when model gets called
```

The inputs for defining our model object are vocab_size and no_of_hidden_units 
(we can decide on actual values for these later). We will need to define an embedding layer
, a single hidden layer with batch normalization, an output layer, a dropout layer, and the loss function.

```
def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
```

Mathematically speaking, there is nothing unique about a word embedding layer. It’s a matrix multiplication with a bias term. A word embedding layer actually has the exact same number of weights as a linear layer. The difference comes down to the PyTorch implementation. The input to the embedding layer is just an index while a linear layer requires an appropriately sized vector. We could use a linear layer if we wanted but we’d have to convert our token IDs to a bunch of 1-hot vectors and do a lot of unnecessary matrix multiplication since most of the entries are 0. As mentioned above, a matrix multiplying a 1-hot vector is the same as just extracting a single column of the matrix, which is a lot faster. Embedding layers are used frequently in NLP since the input is almost always a 1-hot representation.

Even though you will never explicitly see a 1-hot vector in the code, it is very useful intuitively to view the input as a 1-hot vector.

```
def forward(self, x, t):
    
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],t), h[:,0]
```

When we call our model, we will need to provide it with an input x and target labels t.

This implementation is not typical in the sense that a for loop is used for the embedding layer as opposed to the more typical batch processing. Assume the input x is a list of length batch_size and each element of this list is a numpy array containing the token ids for a particular sequence. These sequences are different length which is why x is not simply a torch tensor of size batch_size by sequence_length. Within the loop, the lookup_tensor is a single sequence of token ids which can be fed into the embedding layer. This returns a torch tensor of length sequence_length by embedding_size. We take the mean over the dimension corresponding to the sequence length and append it to the list bow_embedding. This mean operation is considered the bag of words. Note this operation returns the same vector embed regardless of how the token ids were ordered in the lookup_tensor.

The torch.stack() operation is a simple way of converting a list of torch tensors of length embedding_size to a single torch tensor of length batch_size by embedding_size. The bow_embedding is passed through a linear layer, a batch normalization layer, a ReLU operation, and dropout.

Our output layer has only one output even though there are two classes for sentiment (positive and negative). We could’ve had a separate output for each class but I actually saw an improvement of around 1-2% by using the nn.BCEWithLogitsLoss() (binary cross entropy with logits loss) as opposed to the usual nn.CrossEntropyLoss() you usually see with multi-class classification problems.

Note that two values are returned, the loss and the logit score. The logit score can be converted to an actual probability by passing it through the sigmoid function or it can be viewed as any score greater than 0 is considered positive sentiment. The actual training loop ultimately determines what will be done with these two returned values.

BOW_sentiment_analysis.py
This is the file you’ll actually run for training the model.

```
import numpy as np
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

from BOW_model import BOW_model
```

Import all of the basic packages we need as well our BOW_model we defined previously.

```
#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000

x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1
```

Here is where we can much more quickly load in our data one time in the exact format we need. We can also choose our dictionary size at this point. I suggest starting out with 8000 as it was shown earlier (in the preprocessing data section) how this greatly reduces the number of weights in the word embedding layer without ignoring too many unique tokens within the actual dataset. Note that each line within our imdb_train.txt file is a single review made up of token ids. We can convert any token id greater than the dictionary size to the unknown token ID 0. Remember also we actually had 75,000 total training files but only the first 25,000 are labeled (we will use the unlabeled data in part 3). The last three lines of code just grab the first 25,000 sequences and creates a vector for the labels (the first 125000 are labeled 1 for positive and the last 12500 are labeled 0 for negative).

```
x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1
```

Do the same for the test dataset.

```
vocab_size += 1

model = BOW_model(vocab_size,500)
model.cuda()
```

Here we actually define the model with no_of_hidden_units equal to 500. Note I added 1 to vocab_size. Remember we actually added 1 to all of the token ids so we could use id 0 for the unknown token. The code above kept tokens 1 through 8000 as well as 0 meaning the vocab_size is actually 8001.

```
# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
```

Define the optimizer with some desired learning rate.

```
batch_size = 200
no_of_epochs = 6
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []
```

Define some parameters such as the batch_size and no_of_epochs. Put the model in training mode (this allows batch normalization to work correctly as well as dropout). Create a few lists to store any variables of interest.

```
for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input,target)
        loss.backward()

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(x_input,target)
        
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

    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

torch.save(model,'BOW.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data.npy',data)
```

There is nothing particularly unique about this training loop compared to what you’ve seen in the past and it should be pretty self explanatory. The only interesting part is the variable x_input we send to the model is not actually a torch tensor at this moment. It’s simply a list of lists containing the token ids. Remember that this is dealt with in the BOW_model.forward() function.

With the current hyperparameters I’ve provided, this model should achieve around 86%-88% on the test dataset. If you check the original paper that came out with this dataset back in 2011, you’ll see this model performs just as well as nearly all of the techniques shown on page 7.

This should only take about 15-30 minutes to train. Report results for this model as well as at least two other trials (run more but only report on two interesting cases). Try to get a model that overfits (high accuracy on training data, lower on testing data) as well as one that underfits (low accuracy on both). Overfitting is easily achieved by greatly increasing the no_of_hidden_units, removing dropout, adding a second or third hidden layer in BOW_model.py as well as a few other things. Underfitting can be easily done by using significantly less hidden units.

Other things to consider:

- Early stopping - notice how sometimes the test performance will actually get worse the longer you train
- Longer training - this might be necessary depending on the choice of hyperparameters
- Learning rate schedule - after a certain number of iterations or once test performance stops increasing, try reducing the learning rate by some factor
- Larger models may take up too much memory meaning you may need to reduce the batch size
- Play with the dictionary size - see if there is any difference utilizing a large portion of the dictionary (like 100k) compared to very few words (500)

You could try removing stop words, nltk.corpus.stopwords.words('english') returns a list of stop words. You can find their corresponding index using the imdb_dictionary and convert any of these to 0 when loading the data into x_train (remember to add 1 to the index to compensate for the unknown token)
Try SGD optimizer instead of adam. This might take more epochs. Sometimes a well tuned SGD with momentum performs better
Try to develop a basic intuition for selecting these parameters yourself and see how big of an effect you can have on the performance. There are four more vastly different models you will train throughout the rest of this assignment and this intuition will help.
