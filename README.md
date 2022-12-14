# Simple Python Multi GPU Stres Test
by Klemen Kotar

This script will automatically download the CIFAR-10 dataset and train a simple convolutional neural network to classify the images.

The srcipt takes the following command line arguments:
```
 --batch_size: the batch size used - increase this to saturate GPU memory
 --num_workers: the number of workers used to load the data - increase this to test multi core uage
 --num_gpus: the number of gpus used - increase this to test multi GPU training
 --num_epochs: the number of epochs the model will trian for - increase this to make trianing longer
```

To run this you should follow these steps:
```
git clone https://github.com/klemenkotar/multi-gpu-test.git
cd multi-gpu-test
pip install -r requirements.txt
python3 main.py --num_gpus=4 --num_epochs=1000 --num_workers=16 --batch_size=256
```

Training code borrowed from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
