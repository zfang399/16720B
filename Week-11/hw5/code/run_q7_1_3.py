import numpy as np
import scipy.io
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device('cpu')

# Some variables and constants
no_epochs = 20
batch_size = 60
learning_rate = 0.005
momentum = 0.9

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x = np.array([train_x[i, :].reshape((32, 32)) for i in range(train_x.shape[0])])
test_x = np.array([test_x[i, :].reshape((32, 32)) for i in range(test_x.shape[0])])

train_x_ts, train_y_ts = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1), torch.from_numpy(train_y).type(torch.long)
test_x_ts, test_y_ts = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1), torch.from_numpy(test_y).type(torch.long)

train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), shuffle=False)

class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(5*5*16, 100)
        self.fc2 = nn.Linear(100, D_out)

    def forward(self, x):
        output = nn.functional.relu(self.conv1(x))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = nn.functional.relu(self.conv2(output))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = output.view(-1, 5*5*16)
        output = nn.functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output

model = Net(train_x.shape[1], train_y.shape[1])

train_loss = []
train_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
for itr in range(no_epochs):
    total_loss = 0
    total_acc = 0
    for data in train_loader:

        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        # choose the indices of the max element for each example
        targets = torch.max(labels, 1)[1]

        # forward pass
        y_pred = model(inputs)
        predicted_target = torch.max(y_pred, 1)[1]

        loss = nn.functional.cross_entropy(y_pred, targets)
        total_loss += loss.item()
        acc = predicted_target.eq(targets.data).sum().item()/labels.size()[0]
        total_acc += acc

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    total_acc = total_acc/len(train_loader)
    # total_loss = total_loss/batch_size
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

print("Training accuracy: ", train_acc[-1])
fig1 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_acc, c='r', label="train accuracy")
# plt.plot(np.arange(0, no_epochs), valid_acc_list, c='g', label="validation accuracy")
plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
plt.title("Accuracy vs epoch"); plt.legend()
plt.show

fig2 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_loss, c='r', label="train loss")
# plt.plot(np.arange(0, no_epochs), valid_loss_list, c='g', label="validation loss")
plt.xlabel("Epoch number"); plt.ylabel("Loss")
plt.title("Loss vs epoch"); plt.legend()
plt.show()

print(len(test_loader))
test_acc = 0
for data in test_loader:
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])

    targets = torch.max(labels, 1)[1]

    y_pred = model(inputs)
    predicted_target = torch.max(y_pred, 1)[1]

    acc = predicted_target.eq(targets).sum().item()/labels.size()[0]
    test_acc += acc
test_acc = test_acc/len(test_loader)
print("Test accuracy: {:.4f}".format(test_acc))
