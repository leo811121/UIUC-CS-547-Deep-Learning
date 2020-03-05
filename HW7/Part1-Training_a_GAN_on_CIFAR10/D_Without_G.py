import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy as np
from torch.autograd import Variable

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 196,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LayerNorm((196,32,32)),
            nn.LeakyReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm((196,16,16)),
            nn.LeakyReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm((196,16,16)),
            nn.LeakyReLU(),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU(),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU(),
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm((196,4,4)),
            nn.LeakyReLU(),
        )

        self.pool = nn.MaxPool2d(4,4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(196,1),
        ) 
        
        self.fc10 = nn.Sequential(
            nn.Linear(196,10)
        )
        
    def forward(self, x):
        #print('shape',x.size())
        x = self.conv1(x)
        #print('conv1')
        #print('shape',x.size())
        x = self.conv2(x)
        #print('conv2')
        #print('shape',x.size())
        x = self.conv3(x)
        #print('conv3')
        #print('shape',x.size())
        x = self.conv4(x)
        #print('conv4')
        #print('shape',x.size())
        x = self.conv5(x)
        #print('conv5')
        #print('shape',x.size())
        x = self.conv6(x)
        #print('conv6')
        #print('shape',x.size())
        x = self.conv7(x)
        #print('conv7')
        #print('shape',x.size())
        x = self.conv8(x)
        #print('conv8')
        #print('shape',x.size())
        x = self.pool(x)
        #x = x.view(x.size(0),-1)
        x = x.view(-1, 196 * 1 * 1)
        #print('view')
        #print('shape',x.size())
        
        critic = self.fc1(x)
        aux = self.fc10(x) 
        return critic, aux


import numpy as np
import torch
import torchvision
import os 

import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# import data and data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model=discriminator()
#model = torch.load('cifar10.model')
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr=0.0001
model.train()

Epochs = 100
for epoch in range(Epochs):

  if(epoch>6):
           for group in optimizer.param_groups:
                for p in group['params']:
                     state = optimizer.state[p]
                     if 'step' in state.keys():
                          if(state['step']>=1024):
                              state['step'] = 1000

  if(epoch==50):
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr/10.0
  if(epoch==75):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/100.0
  
  for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

      if(Y_train_batch.shape[0] < batch_size):
          continue

      X_train_batch = Variable(X_train_batch).cuda()
      Y_train_batch = Variable(Y_train_batch).cuda()
      _, output = model(X_train_batch)

      loss = criterion(output, Y_train_batch)
      optimizer.zero_grad()

      loss.backward()
      optimizer.step()
  print('train','epoch: ', epoch, 'loss: ',loss)
  
  test_accu = []
  for i, data in enumerate(testloader, 0):
    x_test, y_test = data
    data1, labels1 = Variable(x_test).cuda(), Variable(y_test).cuda()
    optimizer.zero_grad()
    model.eval()
    __, outputs1 = model(data1)     
    prediction = outputs1.data.max(1)[1]  
    accuracy = ( float( prediction.eq(labels1.data).sum() ) /float(batch_size) )*100.0
    test_accu.append(accuracy)
  accuracy = np.mean(test_accu)
  print('test_accu:',accuracy)

 
torch.save(model,'cifar10.model')
