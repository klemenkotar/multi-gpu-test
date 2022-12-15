# Simple Python Multi GPU Stres Test
by Klemen Kotar

This script will automatically download the CIFAR-10 dataset and train a simple convolutional neural network to classify the images.

The srcipt takes the following command line arguments:
 -batch_size: the batch size used - increase this to saturate GPU memory
 -num_workers: the number of workers used to load the data - increase this to test multi core uage
 -num_gpus: the number of gpus used - increase this to test multi GPU training
 -num_epochs: the number of epochs the model will trian for - increase this to make trianing longer
 
Training code borrowed from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
