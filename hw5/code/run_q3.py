import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 80
learning_rate = 0.005
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
no_batches = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, train_y.shape[1], params, "output")

import copy
params_Wlayer1 = copy.deepcopy(params['Wlayer1'])

train_loss_list = []; train_acc_list = []
valid_loss_list = []; valid_acc_list = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!

        h1 = forward(xb, params, "layer1")
        probs = forward(h1, params, "output", softmax)

        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        delta1 = probs - yb
        delta2 = backwards(delta1, params,"output",linear_deriv)
        backwards(delta2, params, "layer1", sigmoid_deriv)

        #Update the gradients
        params["Wlayer1"] -= learning_rate*params["grad_Wlayer1"]
        params["blayer1"] -= learning_rate*params["grad_blayer1"]
        params["Woutput"] -= learning_rate*params["grad_Woutput"]
        params["boutput"] -= learning_rate*params["grad_boutput"]


    total_acc /= no_batches
    # TODO: check whether this is to be done
    total_loss /= no_batches
    train_acc_list.append(total_acc)
    train_loss_list.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    h1 = forward(valid_x, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss_list.append(loss)
    valid_acc_list.append(acc)

fig1 = plt.figure()
plt.plot(np.arange(0, max_iters), train_acc_list, c='r', label="train accuracy")
plt.plot(np.arange(0, max_iters), valid_acc_list, c='g', label="validation accuracy")
plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
plt.title("Accuracy vs epoch"); plt.legend()
plt.show

fig2 = plt.figure()
plt.plot(np.arange(0, max_iters), train_loss_list, c='r', label="train loss")
plt.plot(np.arange(0, max_iters), valid_loss_list, c='g', label="validation loss")
plt.xlabel("Epoch number"); plt.ylabel("Loss")
plt.title("Loss vs epoch"); plt.legend()
plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = valid_acc_list[-1]
print('Validation accuracy: ',valid_acc)

test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ', test_acc)

if False: # view the data
    for crop in xb:
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure()
grid = ImageGrid(fig, 111,
        nrows_ncols=(8, 8), #see 64 weights of 32x32 each
        axes_pad=0.05,
        )
for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32)))
    plt.axis('off')
plt.show()

fig = plt.figure()
grid = ImageGrid(fig, 111,
        nrows_ncols=(8, 8), #see 64 weights of 32x32 each
        axes_pad=0.05,
        )
for i in range(hidden_size):
    grid[i].imshow(np.reshape(params_Wlayer1[:, i], (32, 32)))
    plt.axis('off')
plt.show()

# Q3.1.4
# compute confusion matrix here
import string
def confusionMat(probs, y):
    confusion_matrix = np.zeros((y.shape[1], y.shape[1]))

    y_idx = np.argmax(y, -1)
    pred_y_idx = np.argmax(probs, -1)
    for i in range(y.shape[0]):
        confusion_matrix[y_idx[i], pred_y_idx[i]] += 1
    return confusion_matrix


# Train data
h1 = forward(train_x, params, 'layer1')
train_probs = forward(h1, params, 'output', softmax)
confusion_matrix = confusionMat(test_probs, test_y)

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Confusion matrix for training data')
plt.show()

# Validation data
h1 = forward(valid_x, params, 'layer1')
valid_probs = forward(h1, params, 'output', softmax)
confusion_matrix = confusionMat(valid_probs, valid_y)

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Confusion matrix for validation data')
plt.show()

# Test data
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)
confusion_matrix = confusionMat(test_probs, test_y)

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Confusion matrix for test data')
plt.show()

