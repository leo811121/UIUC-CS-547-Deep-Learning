# 1b: Bag of Words With Using GloVe Features
We will now train another bag of words model but use the pre-trained GloVe features in place of the word embedding layer from before. Typically by leveraging larger datasets and regularization techniques, overfitting can be reduced. The GloVe features were pretrained on over 840 billion tokens. Our training dataset contains 20 million tokens and 2⁄3 of the 20 million tokens are part of the unlabeled reviews which weren’t used in part 1a. The GloVe dataset is over 100 thousand times larger.

The hope is then that these 300 dimensional GloVe features already contain a significant amount of useful information since they were pre-trained on such a large dataset and that will improve performance for our sentiment classification.

Create a directory ‘1b/’. I would go ahead and copy BOW_model.py and BOW_sentiment_analysis.py from part 1a here as most of the code is the same.

BOW_model.py

```
class BOW_model(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.fc_hidden1 = nn.Linear(300,no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        # self.fc_hidden2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        # self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, t):

        h = self.dropout1(F.relu(self.bn_hidden1(self.fc_hidden1(x))))
        # h = self.dropout2(F.relu(self.bn_hidden2(self.fc_hidden2(h))))
        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]
```

This is actually simpler than part 1a considering it doesn’t need to do anything for the embedding layer. We know the input x will be a torch tensor of batch_size by 300 (the mean GloVe features over an entire sequence).

BOW_sentiment_analysis.py

```
glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]

    line = np.mean(glove_embeddings[line],axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]
    
    line = np.mean(glove_embeddings[line],axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = BOW_model(500) # try 300 as well

model.cuda()
```

This first part is nearly the same besides the fact that we can actually go ahead and do the mean operation for the entire sequence one time when loading in the data. We load in the glove_embeddings matrix, convert all out-of-dictionary tokens to the unknown token for each review, extract the embedding for each token in the sequence from the matrix, take the mean of these emeddings, and append this to the x_train or x_test list.

The rest of the code is the same besides grabbing the data for each batch within the actual train/test loop. The code below is for training and you’ll need to modify it slightly for testing by changing x_train to x_test.

```
## within the training loop
        x_input = x_train[I_permutation[i:i+batch_size]]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target)
        loss.backward()
```

Just like before, try a few different hyperparameter settings and report the results.

Against the intuition laid out in the beginning of this section, this model actually seems to perform worse on average than the one in part a. This seems to achieve anywhere between 81-87%.

Let’s take a look at what’s happening. In part 1a, test accuracy typically seems to achieve its max after the 3rd epoch and begins to decrease with more training while the training accuracy continues to increase well into 90+%. This is a sure sign of overfitting. The training accuracy for part 1b stops much earlier (around 86-88%) and doesn’t seem to improve much more.

Nearly 95% of the weights belong to the embedding layer in part 1a. We’re training significantly less in part 1b and can’t actually fine-tune the word embeddings at all. Using only 300 hidden weights for part 1b results in very little overfitting while still achieving decent accuracy.

