# Simple Python Multi GPU Stres Test
# by Klemen Kotar
#
# This script will automatically download the CIFAR-10 dataset and train
# a simple convolutional neural network to classify the images.
# The srcipt takes the following command line arguments:
#   --batch_size: the batch size used - increase this to saturate GPU memory
#   --num_workers: the number of workers used to load the data - increase this to test multi core uage
#   --num_gpus: the number of gpus used - increase this to test multi GPU training
#   --num_epochs: the number of epochs the model will trian for - increase this to make trianing longer
#
# Training code borrowed from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import argparse
import torch
import torchvision
import torchvision.transforms as transforms


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=str, default=8)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-ng', '--num_gpus', type=int, default=0)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    args = parser.parse_args()
    return args


def multi_gpu_test(args):

    # Display Config
    print("Starting GPU Stress Test With The Following Configuration: " +
           "Number of GPUs: {}\nNumber of Workers: {}\nBatch Size: {}\nNumber of Epochs: {}"
           .format(args.num_gpus, args.num_workers, args.batch_size, args.num_epochs))

    # Define image transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define train dataset and dataloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    # Define test dataset and dataloader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

    # Name dataset classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define Simple Convolutional Neural Network
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 6, 5)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(6, 16, 5)
            self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
            self.fc2 = torch.nn.Linear(120, 84)
            self.fc3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = self.pool(torch.nn.functional.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize neural network
    model = Net()

    # Distribute It On The Specified Number of GPUs
    if args.num_gpus == 0:
        device = 'cpu'
    else:
        if args.num_gpus <=  torch.cuda.device_count():
            device = torch.cuda.current_device()
            model = torch.nn.DataParallel(model, 
                device_ids=["cuda:{}".format(i) for i in range(torch.cuda.device_count())]).to(device)
        else:
            raise Exception("Num GPUs specified ({}) is greater than the number available {}".format(
                args.num_gpus, torch.cuda.device_count()
            ))

    # Define Optimizer And Loss Function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train Model
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        # Training Step
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation Step
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    args = get_args()
    multi_gpu_test(args)
