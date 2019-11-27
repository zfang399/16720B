import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
train_mean = np.sum(train_x, axis=0)/train_x.shape[0]
train_x = train_x - train_mean

U, s, Vt = np.linalg.svd(train_x)
# First 32 columns of V matrix
P = Vt[0:32, :]
# rebuild a low-rank version
lrank = train_x @ P.T

# rebuild it
recon = None
recon = lrank @ P
recon += train_mean
train_x += train_mean

recon_valid = None
avg_train_psnr = 0
for i in range(recon.shape[0]):
    avg_train_psnr += psnr(train_x[i, :], recon[i, :])
avg_train_psnr /= recon.shape[0]

# build valid dataset
valid_mean = np.sum(valid_x, axis=0)/valid_x.shape[0]
valid_x = valid_x - valid_mean
recon_valid = (valid_x @ P.T) @ P
recon_valid += valid_mean
valid_x += valid_mean

# visualize the comparison and compute PSNR
avg_valid_psnr = 0
print(recon_valid[0, :].dtype)
print(recon_valid[0, :])
print(valid_x[0, :].dtype)
print(valid_x[0, :])
for i in range(recon_valid.shape[0]):
    avg_valid_psnr += psnr(valid_x[i, :], recon_valid[i, :])
avg_valid_psnr /= recon_valid.shape[0]
print("Average PSNR for validation data: ", avg_valid_psnr)
print("Average PSNR for training data: ", avg_train_psnr)

selected_idx = [1, 10, 101, 111, 201, 211, 301, 311, 401, 411]
for i in selected_idx:
    fig = plt.figure()
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.imshow(valid_x[i, :].reshape((32, 32)).T, cmap='gray')
    ax2.imshow(recon_valid[i, :].reshape((32, 32)).T, cmap='gray')
    plt.show()

