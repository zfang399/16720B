import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

device = torch.device('cuda:0')

train_data = scipy.io.loadmat('../data/nist26_train.mat')
test_data = scipy.io.loadmat('../data/nist26_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
batch_size = 60
learning_rate = 0.005
hidden_size = 64

train_x_ts, train_y_ts = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y).type(torch.long)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True, drop_last=True)

test_x_ts, test_y_ts = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_y).type(torch.long)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=batch_size, shuffle=False)

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = TwoLayerNet(train_x.shape[1], hidden_size, train_y.shape[1])

train_loss = []
train_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
for itr in range(max_iters):
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
plt.plot(np.arange(0, max_iters), train_acc, c='r', label="train accuracy")
# plt.plot(np.arange(0, max_iters), valid_acc_list, c='g', label="validation accuracy")
plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
plt.title("Accuracy vs epoch"); plt.legend()
plt.show

fig2 = plt.figure()
plt.plot(np.arange(0, max_iters), train_loss, c='r', label="train loss")
# plt.plot(np.arange(0, max_iters), valid_loss_list, c='g', label="validation loss")
plt.xlabel("Epoch number"); plt.ylabel("Loss")
plt.title("Loss vs epoch"); plt.legend()
plt.show()

# run on test data
test_acc = 0
for data in test_loader:
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])

    targets = torch.max(labels, 1)[1]

    y_pred = model(inputs)
    predicted_target = torch.max(y_pred, 1)[1]

    loss = nn.functional.cross_entropy(y_pred, targets)
    acc = predicted_target.eq(targets.data).sum().item()/labels.size()[0]
    test_acc += acc

test_acc = test_acc/len(test_loader)
print("Test accuracy: ", test_acc)
