# 2b: With GloVe Features

You should copy RNN_model.py, RNN_sentiment_analysis.py, and RNN_test.py from part 2a over here as a starting point.

RNN_model.py
You’ll still need the LockedDropout() and StatefulLSTM().

```
class RNN_model(nn.Module):
    def __init__(self,no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.lstm1 = StatefulLSTM(300,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        # self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        # self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        # self.lstm2.reset_state()
        # self.dropout2.reset_state()

    def forward(self, x, t, train=True):

        no_of_timesteps = x.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(x[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train)

            # h = self.lstm2(h)
            # h = self.bn_lstm2(h)
            # h = self.dropout2(h,dropout=0.3,train=train)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,features)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps)

        pool = nn.MaxPool1d(x.shape[1])
        h = pool(outputs)
        h = h.view(h.size(0),-1)

        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]#F.softmax(h, dim=1)
```

This is the same model besides the embedding layer is removed from before.

RNN_sentiment_analysis.py
Modify the RNN_sentiment_analysis.py from part 2a to work with GloVe features.

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

    x_train.append(line)
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

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = RNN_model(500)
model.cuda()
```

Load in all of the data.

```
x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 100
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
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        loss.backward()
```

For each batch, make sure to extract the glove_embeddings as shown above before sending it into the model.

RNN_test.py
Here is an example output from one of the models I trained.

```
50 78.94 0.5010 24.4657
100 84.63 0.3956 48.4858
150 87.62 0.3240 75.0972
200 89.18 0.2932 109.8104
250 89.90 0.2750 149.9544
300 90.50 0.2618 193.7136
350 90.83 0.2530 245.5717
400 90.90 0.2511 299.2605
450 91.08 0.2477 360.4909
500 91.19 0.2448 428.9970
```

We see some progress here and the results are the best yet. Although the bag of words model with GloVe features performed poorly, we now see the GloVe features are much better utilized when allowing the model to handle the temporal information and overfitting seems to be less of a problem since we aren’t retraining the embedding layer. As we saw in part 2a, the LSTM seems to be overkill and the dataset doesn’t seem to be complex enough to train a model completely from scratch without overfitting.

