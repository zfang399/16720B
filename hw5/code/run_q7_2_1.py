import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt



train_dir = '../data/oxford-flowers17/train'
test_dir = '../data/oxford-flowers17/test'
validation_dir = '../data/oxford-flowers17/val'

batch_size = 60
num_workers = 4
num_epochs1 = 10
num_epochs2 = 10
learning_rate1 = 0.001
learning_rate2 = 1e-5

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])

val_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

train_dataset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = ImageFolder(validation_dir, transform=val_test_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = ImageFolder(test_dir, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def runEpoch(model, loader, optimizer):
    """Train the model for one epoch

    :model: TODO
    :loader: TODO
    :optimizer: TODO
    :returns: TODO

    """
    model.train()
    total_loss = 0
    for data in loader:
        inputs = torch.autograd.Variable(data[0])
        targets = torch.autograd.Variable(data[1])

        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss

def checkAccuracy(model, loader):
    """Check the accuracy of the model
    :returns: TODO

    """
    model.eval()
    num_correct, num_samples = 0, 0
    for data in loader:
        inputs = torch.autograd.Variable(data[0])
        targets = torch.autograd.Variable(data[1])

        y_pred = model(inputs)
        predicted_target = torch.max(y_pred, 1)[1]
        num_correct += predicted_target.eq(targets).sum().item()
        num_samples += inputs.size(0)
    acc = num_correct/num_samples
    return acc




def squeezenetFinetuning():
    """Finetune and train squeezenet
    :returns: TODO

    """

    # finetune using a single layer classifier using weights from squeezenet1_1 as initialization
    squeezenet_model = torchvision.models.squeezenet1_1(pretrained = True)
    num_classes = len(train_dataset.classes)
    squeezenet_model.classifier[1] = nn.Conv2d(512, num_classes, 1)
    squeezenet_model.num_classes = num_classes

    for param in squeezenet_model.parameters():
        param.requires_grad = False
    for param in squeezenet_model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(squeezenet_model.classifier.parameters(), lr=learning_rate1)
    # Train the last layer for a few epochs
    train_acc_list, val_acc_list = [], []
    for epoch in range(num_epochs1):
        # Run an epoch over training data
        train_loss = runEpoch(squeezenet_model, train_loader, optimizer)

        train_acc = checkAccuracy(squeezenet_model, train_loader)
        val_acc = checkAccuracy(squeezenet_model, val_loader)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if epoch % 2 == 0:
            print("Epoch number: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,train_loss,train_acc))

    for param in squeezenet_model.parameters():
        param.requires_grad = True

    # Construct a new optimizer for the full training
    optimizer = torch.optim.Adam(squeezenet_model.parameters(), lr=learning_rate2)

    # Train the entire model for a few more epochs
    for epoch in range(num_epochs2):
        train_loss = runEpoch(squeezenet_model, train_loader, optimizer)

        train_acc = checkAccuracy(squeezenet_model, train_loader)
        val_acc = checkAccuracy(squeezenet_model, val_loader)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if epoch % 2 == 0:
            print("Epoch number: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch, train_loss, train_acc))

    fig1 = plt.figure()
    no_epochs = num_epochs1 + num_epochs2
    plt.plot(np.arange(0, no_epochs), train_acc_list, c='r', label="train accuracy")
    plt.plot(np.arange(0, no_epochs), val_acc_list, c='b', label="validation accuracy")
    plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
    plt.title("Train Accuracy vs epoch"); plt.legend()
    plt.show()
    # finally check the accuracy on test dataset
    test_acc = checkAccuracy(squeezenet_model, test_loader)
    print("Test accuracy: {:.4f}".format(test_acc))


class Net(nn.Module):

    """Docstring for Net. """

    def __init__(self):
        """TODO: to be defined. """
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                  )
        self.conv2 = nn.Sequential(nn.Conv2d(6, 20, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                  )
        self.conv3 = nn.Sequential(nn.Conv2d(20, 80, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                  )
        self.fc1 = nn.Sequential(nn.Linear(24*24*80, 60),
                                 nn.ReLU(),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(60, 17))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(-1, 24*24*80)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

def scratch():
    """Finetune and train network built from scratch
    :returns: TODO

    """
    num_workers = 4
    num_epochs = 50
    learning_rate = 0.001

    model = Net()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loss, train_acc, val_acc = [], [], []
    for epoch in range(num_epochs):
        total_loss = 0
        no_correct = 0
        for data in train_loader:
            inputs = torch.autograd.Variable(data[0])
            targets = torch.autograd.Variable(data[1])

            y_pred = model(inputs)
            predicted_target = torch.max(y_pred, 1)[1]

            loss = nn.functional.cross_entropy(y_pred, targets)
            total_loss += loss.item()
            no_correct += predicted_target.eq(targets).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = no_correct/len(train_dataset)
        train_loss.append(total_loss)
        train_acc.append(acc)

        valid_acc = checkAccuracy(model, val_loader)
        val_acc.append(valid_acc)
        if epoch % 2 == 0:
            print("Epoch number: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,acc))
        if epoch > 30:
            learning_rate = 0.001/10

    fig1 = plt.figure()
    plt.plot(np.arange(0, num_epochs), train_acc, c='r', label="train accuracy")
    plt.plot(np.arange(0, num_epochs), val_acc, c='b', label="validation accuracy")
    plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
    plt.title("Train Accuracy vs epoch"); plt.legend()
    plt.show()
    # finally check the accuracy on test dataset
    test_acc = checkAccuracy(model, test_loader)
    print("Test accuracy: {:.4f}".format(test_acc))


if __name__ == "__main__":

    squeezenetFinetuning()
    scratch()
