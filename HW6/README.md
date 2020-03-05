# Language Model
## Problem Description
This final section will be about training a language model by leveraging the additional unlabeled data and showing how this pretraining stage typically leads to better results on other tasks like sentiment analysis.

A language model gives some probability distribution over the words in a sequence. We can essentially feed sequences into a recurrent neural network and train the model to predict the following word. Note that this doesn’t require any additional data labeling. The words themselves are the labels. This means we can utilize all 75000 reviews in the training set.

## 3a - Training the Language Model
This section will have two files, RNN_language_model.py and train_language_model.py.

RNN_language_model.py
Once again you will need LockedDropout() and StatefulLSTM().

```
class RNN_language_model(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units):
        super(RNN_language_model, self).__init__()

        
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.lstm3 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm3= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout3 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.decoder = nn.Linear(no_of_hidden_units, vocab_size)

        self.loss = nn.CrossEntropyLoss()#ignore_index=0)

        self.vocab_size = vocab_size

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        self.lstm2.reset_state()
        self.dropout2.reset_state()
        self.lstm3.reset_state()
        self.dropout3.reset_state()

    def forward(self, x, train=True):


        embed = self.embedding(x) # batch_size, time_steps, features

        no_of_timesteps = embed.shape[1]-1

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.3,train=train)

            h = self.lstm2(h)
            h = self.bn_lstm2(h)
            h = self.dropout2(h,dropout=0.3,train=train)

            h = self.lstm3(h)
            h = self.bn_lstm3(h)
            h = self.dropout3(h,dropout=0.3,train=train)

            h = self.decoder(h)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,vocab_size)
        target_prediction = outputs.permute(1,0,2) # batch, time, vocab
        outputs = outputs.permute(1,2,0) # (batch_size,vocab_size,time_steps)

        if(train==True):

            target_prediction = target_prediction.contiguous().view(-1,self.vocab_size)
            target = x[:,1:].contiguous().view(-1)
            loss = self.loss(target_prediction,target)

            return loss, outputs
        else:
            return outputs
```

Unlike before, the final layer has more outputs (called decoder) and we no longer do any sort of pooling. Each output of the sequence will be used separately for calculating a particular loss and all of the losses within a sequence will be summed up. The decoder layer has an input dimension the same size as no_of_hidden_states and the output size is the same as the vocab_size. After every timestep, we have an output for a probability distribution over the entire vocabulary.

After looping through from i=0 to i=no_of_timesteps-1, we have outputs for i=1 to i=no_of_timesteps stored in target_prediction. Notice the variable target is simply the input sequence x[:,1:] without the first index.

Lastly, we’ll start off with three stacked LSTM layers. The task of predicting the next word is far more complicated than predicting sentiment for the entire phrase. We don’t need to be as worried about overfitting since the dataset is larger (although it’s still an issue).

train_language_model.py
Copy RNN_sentiment_analysis.py from part 2a.

First make sure to use all the training data including the unlabeled ones by removing the following lines:

```
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1
...
L_Y_train = len(y_train)
```

and change these two lines:

```
I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):
```

into

```
 I_permutation = np.random.permutation(len(x_train))

    for i in range(0, len(x_train), batch_size):
 ```
 
Also make sure to import the RNN_language_model you made above.

```
print('begin training...')
for epoch in range(0,75):

    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR/10.0

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl<sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = Variable(torch.LongTensor(x_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input)
        loss.backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(),2.0)

        optimizer.step()   # update gradients
        
        values,prediction = torch.max(pred,1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
        epoch_acc += accuracy
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
        
        if (i+batch_size) % 1000 == 0 and epoch==0:
           print(i+batch_size, accuracy/batch_size, loss.data.item(), norm, "%.4f" % float(time.time()-time1))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    ## test
    if((epoch+1)%1==0):
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            sequence_length = 100
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl<sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            x_input = Variable(torch.LongTensor(x_input)).cuda()

            with torch.no_grad():
                pred = model(x_input,train=False)
            
            values,prediction = torch.max(pred,1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
            epoch_acc += accuracy
            epoch_loss += loss.data.item()
            epoch_counter += batch_size
            #train_accu.append(accuracy)
            if (i+batch_size) % 1000 == 0 and epoch==0:
               print(i+batch_size, accuracy/batch_size)
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))
    torch.cuda.empty_cache()

    if(((epoch+1)%2)==0):
        torch.save(model,'temp.model')
        torch.save(optimizer,'temp.state')
        data = [train_loss,train_accu,test_accu]
        data = np.asarray(data)
        np.save('data.npy',data)
torch.save(model,'language.model')
```

Gradient clipping is added in this training loop. Recurrent neural networks can sometimes experience extremely large gradients for a single batch which can cause them to be difficult to train without the gradient clipping.

This model takes much longer to train, about a day. The accuracy will be relatively low (which makes sense considering it’s trying to predict one of 8000 words) but this doesn’t actually tell you much. It’s better to go by the loss. Perplexity is typically used when comparing language models. More about how the cross entropy loss and perplexity are related can be read about here.

I trained multiple models with various results. The model here trained for 75 epochs with a sequence length of 50.

## 3b - Generating Fake Reviews

Although a general language model assigns a probability P(w_0, w_1, …, w_n) over the entire sequence, we’ve actually trained ours to predict P(w_n|w_0, …, w_n-1) where each output is conditioned only on previous inputs. This gives us the ability to sample from P(w_n|w_0, …, w_n-1), feed this sampled token back into the model, and repeat this process in order to generate fake movie reviews. This section will walk you through this process.

Make a new file called generate_review.py.

```
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist

import h5py
import time
import os
import io

import sys

from RNN_language_model import RNN_language_model

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}
```

We will actually utilize the vocabulary we constructed earlier to convert sampled indices back to their corresponding words.

```
model = torch.load('language.model')
print('model loaded...')
model.cuda()

model.eval()
```

Load the trained language model from part 3a and set it to eval() mode.

```
## create partial sentences to "prime" the model
## this implementation requires the partial sentences
## to be the same length if doing more than one
# tokens = [['i','love','this','movie','.'],['i','hate','this','movie','.']]
tokens = [['a'],['i']]

token_ids = np.asarray([[word_to_id.get(token,-1)+1 for token in x] for x in tokens])

## preload phrase
x = Variable(torch.LongTensor(token_ids)).cuda()

embed = model.embedding(x) # batch_size, time_steps, features

state_size = [embed.shape[0],embed.shape[2]] # batch_size, features
no_of_timesteps = embed.shape[1]

model.reset_state()

outputs = []
for i in range(no_of_timesteps):

    h = model.lstm1(embed[:,i,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    h = model.decoder(h)

    outputs.append(h)

outputs = torch.stack(outputs)
outputs = outputs.permute(1,2,0)
output = outputs[:,:,-1]
```

We can start sampling at the very start or after the model has processed a few words already. The latter is akin to autocomplete. In this example, I’m generating two reviews. The first starts simply with the letter/word ‘a’ and the second starts with the letter/word ‘i’. These are both stored in tokens and converted to token_ids in order to be used as the inputs.

The bottom portion of code then loops through the sequences (both sequences at the same time using batch processing) and “primes” the model with our partial sentences. The variable output will be size batch_size by vocab_size. Remember this output is not a probability. After passing it through the softmax function, we can interpret it as a probability and sample from it.

```
temperature = 1.0 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1)
    review.append(x.cpu().data.numpy()[:,0])

    ## predict the next word
    embed = model.embedding(x)

    h = model.lstm1(embed)
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    output = model.decoder(h)
```

Here is where we will actually generate the fake reviews. We use the previous output, perform the softmax function (assign probability of 0.0 to token id 0 to ignore the unknown token), randomly sample based on probs, save these indices to the list review, and finally get another output.

```
review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)
```

Here we simply convert the token ids to their corresponding string. Remember all of the indices need -1 to account for the unknown token we added before using it with imdb_dictionary.

```
# temperature 1.0
a hugely influential , very strong , nuanced stand up comedy part all too much . this is a film that keeps you laughing and cheering for your own reason to watch it . the same has to be done with actors , which is surely `` the best movie '' in recent history because at the time of the vietnam war , ca n't know who to argue they claim they have no choice ... out of human way or out of touch with personal history . even during the idea of technology they are not just up to you . there is a balance between the central characters and even the environment and all of that . the book is beautifully balanced , which is n't since the book . perhaps the ending should have had all the weaknesses of a great book but the essential flaw of the 

i found it fascinating and never being even lower . spent nothing on the spanish 's particularly good filming style style . as is the songs , there 's actually a line the film moves so on ; a sequence and a couple that begins almost the same exact same so early style of lot of time , so the indians theme has not been added to the on screen . well was , the movie has to be the worst by the cast . i did however say - the overall feel of the film was superb , and for those that just do n't understand it , the movie deserves very little to go and lets you see how it takes 3 minutes to chilling . i must admit the acting was adequate , but `` jean reno '' was a pretty good job , he was very subtle 
```

Although these reviews as a whole don’t make a lot of sense, it’s definitely readable and the short phrases seem quite realistic. The temperature parameter from before essentially adjusts the confidence of the model. Using temperature=1.0 is the same as the regular softmax function which produced the reviews above. As the temperature increases, all of the words will approach having the same probability. As the temperature decreases, the most likely word will approach a probability of 1.0.

```
# temperature 1.3
a crude web backdrop from page meets another eastern story ( written by an author bought ) when it was banned months , i am sure i truly are curiosity ; i have always been the lone clumsy queen of the 1950 's shoved director richard `` an expert on target '' . good taste . not anything report with star 70 's goods , having it worked just equally attractive seem to do a moving train . memorable and honest in the case of `` ross , '' you find it fantasy crawford is strong literature , job suffering to his a grotesque silent empire , for navy turns to brooklyn and castle of obsession that has already been brought back as welles ' anthony reaches power . it 's totally clearly staged , a cynical sit through a change unconscious as released beer training in 1944 with mickey jones 

i wanted to walk out on featuring this expecting even glued and turd make . he genius on dialog , also looking good a better opportunity . anyway , the scene in the ring where christopher wallace , said things giving the least # 4 time anna hang earlier too leaves rick the blond doc from , walter from leon . is ironic until night with rob roy , he must 've been a mother . which are images striking to children while i think maybe this is not mine . but not in just boring bull weather sake , which set this by saying an excellent episode about an evil conspiracy monster . minor character with emphasis for blood deep back and forth between hip hop , baseball . as many red light figure hate americans like today 's life exercise around the british variety kids . nothing was added 
```

Note here with a higher temperature, there is still some sense of structure but the phrases are very short and anything longer than a few words doesn’t begin to make much sense. Choosing an even larger temperature would result in random words being chosen from the dictionary.

```
## temperature 0.25
a little slow and i found myself laughing out loud at the end . the story is about a young girl who is trying to find a way to get her to go to the house and meets a young girl who is also a very good actress . she is a great actress and is very good . she is a great actress and she is awesome in this movie . she is a great actress and i think she is a great actress . i think she is a great actress and i hope she will get more recognition . i think she has a great voice . i think she is a great actress . i think she is one of the best actresses in the world . she is a great actress and i think she is a great actress . she is a great actress and

i was a fan of the original , but i was very disappointed . the plot was very weak , the acting was terrible , the plot was weak , the acting was terrible , the story was weak , the acting was horrible , the story was weak , the acting was bad , the plot was bad , the story was worse , the acting was bad , the plot was stupid , the acting was bad , the plot was stupid and the acting was horrible . i do n't know how anyone could have made this movie a 1 . i hope that it will be released on dvd . i hope it gets released on dvd soon . i hope that it will be released on dvd . i hope it gets released soon because it is so bad it 's good . i hope that
```

With a lower temperature, the predictions can get stuck in loops. However, it is interesting to note how the top review here happened to be a “good” review while the bottom review happened to be a “bad” review. It seems that once the model becomes confident with the tone of the review, it sticks with it. Remember that this language model was trained to simply predict the next word in 12500 positive reviews, 12500 negative reviews, and 50000 neutral reviews. It seems to naturally be taking into consideration the sentiment without explicitly being told to do so.

```
# temperature = 0.5
a very , very good movie , but it is a good movie to watch . the plot is great , the acting is good , the story is great , the acting is good , the story is good , there is plenty of action . i can not recommend this movie to anyone . i would recommend it to anyone who enjoys a good movie . i 'm not sure that the movie is not a good thing , but it is a good movie . it is a great movie , and a must see for anyone who has n't seen it . the music is great , the songs are great , and the music is fantastic . i do n't think i have ever seen a movie that is so good as the story line . i would recommend this movie to anyone who wants to 

i usually like this movie , but this is one of those movies that i could n't believe . i was so excited that the movie was over , i was very disappointed . it was n't that bad , but it was n't . it was n't even funny . i really did n't care about the characters , and the characters were so bad that i could n't stop laughing . i was hoping for a good laugh , but i was n't expecting much . the acting was terrible , the acting was poor , the story moved , and the ending was so bad it was just so bad . the only thing that kept me from watching this movie was the fact that it was so bad . i 'm not sure if it was a good movie , but it was n't . i was 
```

We’re not necessarily generating fake reviews here for any particular purpose (although there are applications that call for this). This is more to simply show what the model learned. It’s simple enough to see the accuracy increasing and the loss function decreasing throughout part 3a, but this helps us get a much better intuitive understanding of what the model is focusing on.

Create some of your own movie reviews with your trained model. The temperature parameter can be sensitive depending on how long the model was trained for. Try “priming” the model with longer phrases (“I hate this movie.”/“I love this movie.”) to see if the sentiment is maintained throughout the generated review.

## 3c - Learning Sentiment
After getting a basic understanding in part 3b of what sort of information the language model has captured so far, we will use it as a starting point for training a sentiment analysis model.

Copy all your code from part 2a. Modify RNN_model.py to have three lstm layers just like your language model. There are two small differences to be made in RNN_sentiment_analysis.py.

```
model = RNN_model(vocab_size,500))

language_model = torch.load('language.model')
model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())
model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())
model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())
model.cuda()
```

You are creating a new RNN_model for sentiment analysis but copying all of the weights for the embedding and LSTM layers from the language model.

```
params = []
# for param in model.embedding.parameters():
#     params.append(param)
# for param in model.lstm1.parameters():
#     params.append(param)
# for param in model.bn_lstm1.parameters():
#     params.append(param)
# for param in model.lstm2.parameters():
#     params.append(param)
# for param in model.bn_lstm2.parameters():
#     params.append(param)
for param in model.lstm3.parameters():
    params.append(param)
for param in model.bn_lstm3.parameters():
    params.append(param)
for param in model.fc_output.parameters():
    params.append(param)

opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(params, lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(params, lr=LR, momentum=0.9)
```

Before defining the optimizer, we’re going to make a list of parameters we want to train. The model will overfit if you train everything (you can and should test this yourself). However, you can choose to just fine-tune the last LSTM layer and the output layer.

Train and test the model now just as you did in part 2a.

Here is an example output of mine for a particular model:

```
50 80.98 0.4255 17.0083
100 87.17 0.3043 30.2916
150 90.18 0.2453 45.0554
200 91.15 0.2188 59.9038
250 91.96 0.2022 74.8118
300 92.34 0.1960 89.7251
350 92.64 0.1901 104.7904
400 92.83 0.1863 119.8761
450 92.95 0.1842 134.8656
500 93.01 0.1828 150.3047
```

This performs better than all of the previous models. By leveraging the additional unlabeled data and pre-training the network as a language model, we can achieve even better results than the GloVe features trained on a much larger dataset. To put this in perspective, the state of the art on the IMDB sentiment analysis is around 97.4%.

## Result


