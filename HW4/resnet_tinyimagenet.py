import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import os

batch_size = 128

def create_val_folder(val_dir):
  """
  This method is responsible for separating validation
  images into separate sub folders
  """
  # path where validation data is present now
  path = os.path.join(val_dir, 'images')
  # file where image2class mapping is present
  filename = os.path.join(val_dir, 'val_annotations.txt')
  fp = open(filename, "r") # open file in read mode
  data = fp.readlines() # read line by line
  ''' 
  Create a dictionary with image names as key and
  corresponding classes as values
  '''
  val_img_dict = {}
  for line in data:
     words = line.split("\t")
     val_img_dict[words[0]] = words[1]
  fp.close()
  # Create folder if not present, and move image into proper folder
  for img, folder in val_img_dict.items():
     newpath = (os.path.join(path, folder))
     if not os.path.exists(newpath): # check if folder exists
         os.makedirs(newpath)
     # Check if image exists in default directory
     if os.path.exists(os.path.join(path, img)):
         os.rename(os.path.join(path, img), os.path.join(newpath, img))
  return

transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(20),
     transforms.RandomCrop(64, padding=4),
    # transforms.RandomGrayscale(p=0.1),
     transforms.ToTensor(),
    ])

# Your own directory to the train folder of tiyimagenet
train_dir = '/u/training/tra335/scratch/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
val_dir = '/u/training/tra335/scratch/tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

'''
train_dir = '/u/training/tra335/scratch/tiny-imagenet-200/train'
train_dataset = datasets.ImageFolder(train_dir,
    transform=transform_train)
print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=8)
val_dir = '/u/training/tra335/scratch/tiny-imagenet-200/val/'
if 'val_' in os.listdir(val_dir)[0]:
    create_val_folder(val_dir)
else:
    pass
val_dataset = datasets.ImageFolder(val_dir,
    transform=transforms.ToTensor())
print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=batch_size, shuffle=False, num_workers=8)
'''

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
  def __init__(self, block, dulplicates, num_classes=200):
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
          nn.Dropout2d(p = 0.01)
        )
    self.in_channels = 32
    self.conv2_x = self.make_layer(block, dulplicate[0], out_channels = 32, stride = 1)
    self.conv3_x = self.make_layer(block, dulplicate[1], out_channels = 64, stride = 2)
    self.conv4_x = self.make_layer(block, dulplicate[2], out_channels = 128, stride = 2)
    self.conv5_x = self.make_layer(block, dulplicate[3], out_channels = 256, stride = 2)
    
    self.maxpool = nn.AvgPool2d(kernel_size=4, stride=1)
    self.fc = nn.Linear(6400, num_classes)
  
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
    x = self.conv1(x)
   # print('conv1',x.size())
    x = self.conv2_x(x)
   # print('conv2',x.size()) 
    x = self.conv3_x(x)
   # print('conv3',x.size())
    x = self.conv4_x(x)
   # print('conv4',x.size()) 
    x = self.conv5_x(x)
   # print('conv5',x.size()) 
    x = self.maxpool(x)
   # print('max',x.size())
    x = x.view(x.shape[0], -1)
   # print('flatten',x.size())
    x = self.fc(x)
   # print('fc')
    return x

dulplicate = [2,4,4,2]
model = ResNet(BasicBlock, dulplicate)
model.cuda()

Path_Save = '/u/training/tra335/Models_Tiny'
torch.save(model.state_dict(), Path_Save)
#model.load_state_dict(torch.load(Path_Save))

# define optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-4, weight_decay=1e-4)
#optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

#Learning Rate Schedule
torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

#Train Model
model.train()
batch_size = batch_size
num_epochs = 100
for epoch in range(num_epochs):
    print('epoch:',epoch)  
    train_accu = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, data in enumerate(train_loader, 0):
        x_train, y_train = data
        data, labels = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        model.train()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        if(epoch>6):
           for group in optimizer.param_groups:
                for p in group['params']:
                     state = optimizer.state[p]
                     if 'step' in state.keys():
                          if(state['step']>=1024):
                              state['step'] = 1000
        optimizer.step()
        prediction = outputs.data.max(1)[1]  
        accuracy = ( float( prediction.eq(labels.data).sum() ) /float(batch_size) )
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch,accuracy_epoch)
    test_accu = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, data in enumerate(val_loader, 0):
      x_test, y_test = data
      data1, labels1 = x_test.to(device), y_test.to(device)
      optimizer.zero_grad()
      model.eval()
      outputs1 = model(data1)     
      prediction = outputs1.data.max(1)[1]  
      accuracy = ( float( prediction.eq(labels1.data).sum() ) /float(batch_size) )
      test_accu.append(accuracy)
    accuracy = np.mean(test_accu)
    print('test_accu:',accuracy)