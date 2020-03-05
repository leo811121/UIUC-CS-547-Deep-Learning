# HW4
## Problem Description
Implement a deep residual neural network for CIFAR100. 
In order to get a full credit, you need to achieve at least the accuracy mentioned below for
each task except for ResNet CIFAR100 with asynchronous SGD. ResNet CIFAR100 with
asynchronous SGD is a 10-point bonus question.
|       File      | Accuracy |
| ----------------| ---------|
| ResNet CIFAR100 |   50%    |
| ResNet CIFAR100 |   70%    |
| ResNet TinyImageNet |   60%|
| ResNet TinyImageNet |   50%|
| Pre-trained ResNet CIFAR100|  70%|
| ResNet CIFAR100 with synchronous SGD |  60% |
## Implementation
### Resnet Architecture
<div align=center>
	<img src = "https://github.com/leo811121/UIUC-CS-547-Deep-Learning/blob/master/HW4/Image/Resnet_2.PNG" width="400">
  <img src = "https://github.com/leo811121/UIUC-CS-547-Deep-Learning/blob/master/HW4/Image/Resnet_1.png" width="400">
</div>

### Distributed Training
Using distributed training to implement Synchronous SGD method on ResNet with multicore
<div align=center>
	<img src = "https://github.com/leo811121/UIUC-CS-547-Deep-Learning/blob/master/HW4/Image/Sync_alg.png" width="400">
  <img src = "https://github.com/leo811121/UIUC-CS-547-Deep-Learning/blob/master/HW4/Image/All_reduce.PNG" width="400">
</div>
## Result

