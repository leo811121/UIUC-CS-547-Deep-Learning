import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
#from torchvision import transforms
#from PIL import Image
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor



# import data and data augmentation
batch_size = 128
transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(20),
     transforms.RandomGrayscale(p=0.1),
     transforms.ToTensor(),
     transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
transform1 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
#Basic Block
class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""

    def __init__(self, in_channels, out_channels, strides, downsample):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        # First Stack
        self.conv1 = nn.Sequential(
          nn.Conv2d( in_channels,
                     out_channels,
                     kernel_size=3,
                     stride = strides,
                     padding=1,
                     bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU()
        )
        
        #Second Stack
        self.conv2 = nn.Sequential(
          nn.Conv2d( out_channels,
                     out_channels,
                     kernel_size=3,
                     stride = 1,
                     padding=1,
                     bias=False),
          nn.BatchNorm2d(out_channels)
        )

        self.dim_change = nn.Sequential(
                     nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=strides,
                              padding=0,
                              bias=False),
                     nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if (self.downsample != False):
          residual = self.dim_change(residual)
        x += residual
        return x
          


class ResNet(nn.Module):
  def __init__(self, block, dulplicates, num_classes=100):
    super(ResNet, self).__init__()
    
    self.conv1 = nn.Sequential(
          nn.Conv2d( in_channels= 3 ,
                     out_channels=32,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout2d(p = 0.1)
        )
    self.in_channels = 32
    self.conv2_x = self.make_layer(block, dulplicate[0], out_channels = 32, stride = 1)
    self.conv3_x = self.make_layer(block, dulplicate[1], out_channels = 64, stride = 2)
    self.conv4_x = self.make_layer(block, dulplicate[2], out_channels = 128, stride = 2)
    self.conv5_x = self.make_layer(block, dulplicate[3], out_channels = 256, stride = 2)
    
    self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
    
    self.fc = nn.Linear(256, num_classes)
  
  def Downsample_Judge(self, stride, out_channels):
    if (stride != 1 or (self.in_channels != out_channels)):
      return True
    else:
      return False
 
  # makeing layer
  def make_layer(self, block, dulplicate, out_channels, stride):
    
    downsample = self.Downsample_Judge(stride, out_channels)
    layers = []
    layers.append(block(self.in_channels, out_channels, stride, downsample))
 
    self.in_channels = out_channels
    
    for dul in range(dulplicate-1):
      layers.append(block(out_channels, out_channels, strides=1, downsample = False))
    WholeStack = nn.Sequential(*layers)
    
    return WholeStack
  
  #forward method
  def forward(self, x):
    #print('x',x.size())
    x = self.conv1(x)
    #print('conv1',x.size())
    x = self.conv2_x(x)
    #print('conv2',x.size())
    x = self.conv3_x(x)
    #print('conv3',x.size())
    x = self.conv4_x(x)
    #print('conv4',x.size())
    x = self.conv5_x(x)
    #print('conv5',x.size())
    x = self.maxpool(x)
    #print('max',x.size())
    x = x.view(x.shape[0], -1)
    #print('flatten',x.size())
    x = self.fc(x)
    
    return x

dulplicate = [2,4,4,2]
model = ResNet(BasicBlock, dulplicate)

# Make sure that all nodes have the same model
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

model.cuda()

Path_Save = '/u/training/tra335/sync_sgd_cifar100/Models'
torch.save(model.state_dict(), Path_Save)
#model.load_state_dict(torch.load(Path_Save))

# define optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

#Train Model

num_epochs = 100
batch_size = float(batch_size)
for epoch in range(num_epochs):  
    train_accu = []
    for step, (x_train, y_train) in enumerate(trainloader, 0):
        data, labels = Variable(x_train).cuda(), Variable(y_train).cuda()
        optimizer.zero_grad()
        model.train()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()

        for param in model.parameters():
            #print(param.grad.data)
            try:
                tensor0 = param.grad.data.cpu()
                dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
                tensor0 /= float(num_nodes)
                param.grad.data = tensor0.cuda()
            except:
                pass

        optimizer.step()
        prediction = outputs.data.max(1)[1]  
        accuracy = ( float( prediction.eq(labels.data).sum() ) / batch_size )
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print('epoch:', epoch,'train accuracy', accuracy_epoch)
    test_accu = []
    for step, (x_test, y_test) in enumerate(testloader, 0):
      data1, labels1 = Variable(x_test).cuda(), Variable(y_test).cuda()
      optimizer.zero_grad()
      model.eval()
      outputs1 = model(data1)     
      prediction = outputs1.data.max(1)[1]  
      accuracy = ( float( prediction.eq(labels1.data).sum() ) /batch_size )
      test_accu.append(accuracy)
    accuracy = np.mean(test_accu)
    print('test accuracy',accuracy)
   # if (accuracy>=0.6):
   #  break