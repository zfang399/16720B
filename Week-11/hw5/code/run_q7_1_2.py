import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cpu')

# Some variables and constants
no_epochs = 4
batch_size = 60
learning_rate = 0.005
momentum = 0.9

transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
                ])
# Load the training dataset
train_loader = DataLoader(torchvision.datasets.MNIST('../data/', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)

test_loader = DataLoader(torchvision.datasets.MNIST('../data/', train=False, download=True, transform=transform),
                         shuffle=False)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*20, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        output = nn.functional.relu(self.conv1(x))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = nn.functional.relu(self.conv2(output))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = output.view(-1, 4*4*20)
        output = nn.functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output

model = Net()

train_loss = []
train_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
for itr in range(no_epochs):
    total_loss = 0
    total_acc = 0
    for data in train_loader:

        inputs = torch.autograd.Variable(data[0])
        targets = torch.autograd.Variable(data[1])

        y_pred = model(inputs)
        predicted_target = torch.max(y_pred, 1)[1]

        loss = nn.functional.cross_entropy(y_pred, targets)
        total_loss += loss.item()
        acc = predicted_target.eq(targets.data).sum().item()/targets.size()[0]
        total_acc += acc

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    total_acc = total_acc/len(train_loader)
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

print("Training accuracy: ", train_acc[-1])
fig1 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_acc, c='r', label="train accuracy")
plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
plt.title("Accuracy vs epoch"); plt.legend()
plt.show

fig2 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_loss, c='r', label="train loss")
plt.xlabel("Epoch number"); plt.ylabel("Loss")
plt.title("Loss vs epoch"); plt.legend()
plt.show()

# run on test data
test_acc = 0
for data in test_loader:
    inputs = torch.autograd.Variable(data[0])
    targets = torch.autograd.Variable(data[1])

    y_pred = model(inputs)
    predicted_target = torch.max(y_pred, 1)[1]

    loss = nn.functional.cross_entropy(y_pred, targets)
    acc = predicted_target.eq(targets.data).sum().item()/targets.size()[0]
    test_acc += acc

test_acc = test_acc/len(test_loader)
print("Test accuracy: ", test_acc)
