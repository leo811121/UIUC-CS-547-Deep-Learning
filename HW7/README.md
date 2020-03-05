# Understanding CNNs and Generative Adversarial Networks

The assignment consists of training a Generative Adversarial Network on the CIFAR10 dataset as well as a few visualization tasks for better understanding how a CNN works.

-Train a baseline model for CIFAR10 classification (~2 hours training time)
- Train a discriminator/generator pair on CIFAR10 dataset utilizing techniques from ACGAN and Wasserstein GANs (~40-45 hours training time)
- Use techniques to create synthetic images maximizing class output scores or particular features as a visualization technique to understand how a CNN is working (<1 minute)
- This homework is slightly different than previous homeworks as you will be provided with a lot of code and sets of hyperparameters. You should only need to train the models once after you piece together the code and understand how it works. The second part of the homework (feature visualization) utilizes the models trained from part 1. The output will need to be run many times with various parameters but the code is extremely fast.

The assignment is written more like a tutorial. The Background section is essentially a written version of the lecture and includes links to reference materials. It is very important to read these. Part 1 will take you through the steps to train the necessary models. Part 2 will take you through the steps of getting output images which are ultimately what will be turned in for a grade. Parts 1 and 2 will make frequent references to the background section.

Make sure to load the more recent pytorch version. Some of the code may not work if you learn the default version on BlueWaters

## Background
### Discriminative vs Generative Models
A generative method attempts to model the full joint distribution P(X,Y). For classification problems, we want the posterior probability P(Y|X) which can be gotten from the modeled P(X,Y) with Bayes' Rule. In a similar fashion, we can also get the likelihood P(X|Y).

As an example, let X be a random variable for shoe size with Y being a label from {male,female}. Shoe size is a multi-modal distribution that can be modeled as two separate Gaussian distributions: one for male and one for female.

