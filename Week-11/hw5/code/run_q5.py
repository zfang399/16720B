import numpy as np
import scipy.io
from nn import *
from collections import Counter


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')
keys = list(params.keys())
for key in keys:
    # Only weight and bias layers do not have '_'
    if '_' not in key:
        params['m_'+key] = np.zeros(params[key].shape)

# should look like your previous training loops
training_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden1', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        total_loss += np.sum(np.square(out - xb))

        delta1 = 2*(out - xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden1', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)

        # Update the weights with gradients
        for key in params.keys():
            if '_' not in key:
                params['m_' + key] = 0.9*params['m_' + key] - learning_rate*params['grad_' + key]
                params[key] += params['m_'+key]

    total_loss /= batch_size
    training_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

import matplotlib.pyplot as plt
fig1 = plt.figure()
plt.plot(np.arange(0, max_iters), training_loss)
plt.xlabel('Epoch number'); plt.ylabel('Average loss')
plt.title('Average loss over epochs')
plt.show()
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
# Forward pass over validation data set
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden1', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)

selected_idx = [1, 10, 101, 111, 201, 211, 301, 311, 401, 411]
for i in selected_idx:
        fig2 = plt.figure()
        ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
        input_img = np.reshape(valid_x[i, :], (32, 32)).T
        compressed_img = np.reshape(out[i, :], (32, 32)).T
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(compressed_img, cmap='gray')
        plt.show()
# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
avg_psnr = 0
for i in range(valid_x.shape[0]):
    avg_psnr += psnr(valid_x[i, :].reshape((32, 32)), out[i, :].reshape((32, 32)))
avg_psnr = avg_psnr/valid_x.shape[0]

print("Average Peak Signal to Noise Ratio: ", avg_psnr)
