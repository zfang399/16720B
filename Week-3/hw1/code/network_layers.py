import numpy as np
import scipy.ndimage
import os
import timeit
import skimage

def extract_deep_feature(x, vgg16_weights):
        '''
        Extracts deep features from the given VGG-16 weights.

        [input]
        * x: numpy.ndarray of shape (H, W, 3)
        * vgg16_weights: list of shape (L, 3)

        [output]
        * feat: numpy.ndarray of shape (K)
        '''
        # resize the image to 224x224 as expected by VGG network
        if np.max(x) > 1.0:
            x = x.astype('float')/255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = skimage.transform.resize(x, (224, 224, 3))
        x = (x - mean)/std

        counter = 2 # counter to stop the convolution at the second linear layer
        starttime = timeit.default_timer()
        vgg16_weights = vgg16_weights[:-2]
        for layer in vgg16_weights:
            print(layer[0])
            if layer[0] == 'conv2d':
                out = multichannel_conv2d(x, layer[1], layer[2])
            if layer[0] == 'relu':
                out = relu(x)
            if layer[0] == 'maxpool2d':
                out = max_pool2d(x, layer[1])
            if layer[0] == 'linear':
                if x.ndim > 1:
                    x = x.transpose(2, 0, 1)
                out = linear(x, layer[1], layer[2])

            x = out
            print(x.shape)

        elapsed = timeit.default_timer() - starttime
        print("Time taken = {}".format(elapsed))
        return x

def multichannel_conv2d(x, weight, bias):
        '''
        Performs multi-channel 2D convolution.

        [input]
        * x: numpy.ndarray of shape (H, W, input_dim)
        * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
        * bias: numpy.ndarray of shape (output_dim)

        [output]
        * feat: numpy.ndarray of shape (H, W, output_dim)
        '''

        # flip the kernel as pytorch vgg16 internally uses correlation against convolution
        weight = np.flip(weight, axis = 2)
        weight = np.flip(weight, axis = 3)
        H, W, input_dim = x.shape
        output_dim, _, kernel_size, _ = weight.shape

        feat = np.empty((H, W, 0))
        for out_channel in range(0, output_dim):
            channel_feature = np.zeros((H, W))
            for channel in range(0, input_dim):
                channel_feature += scipy.ndimage.convolve(x[:, :, channel], weight[out_channel, channel, :, :],
                                                        mode='constant', cval=0.0)
            channel_feature = channel_feature + bias[out_channel]
            feat = np.dstack((feat, channel_feature))

        print(feat.shape)
        return feat

def relu(x):
        '''
        Rectified linear unit.

        [input]
        * x: numpy.ndarray

        [output]
        * y: numpy.ndarray
        '''
        y = np.maximum(0, x)
        return y

def max_pool2d(x, size):
        '''
        2D max pooling operation.

        [input]
        * x: numpy.ndarray of shape (H, W, input_dim)
        * size: pooling receptive field

        [output]
        * y: numpy.ndarray of shape (H/size, W/size, input_dim)
        '''
        H, W, input_dim = x.shape
        # fix the output size
        subsampled_h = H//size
        subsampled_w = W//size
        padded_x = x[:subsampled_h*size, :subsampled_w*size, ...]

        # reshape the input matrix into a 4 dimension and then find the max
        # across the 2nd and 4th dimension to obtain a output of (subsampled_h, subsampled_w, input_dim)
        new_shape = (subsampled_h, size, subsampled_w, size, input_dim)
        y = np.amax(padded_x.reshape(new_shape), axis=(1, 3))

        return y

def linear(x,W,b):
        '''
        Fully-connected layer.

        [input]
        * x: numpy.ndarray of shape (input_dim)
        * weight: numpy.ndarray of shape (output_dim,input_dim)
        * bias: numpy.ndarray of shape (output_dim)

        [output]
        * y: numpy.ndarray of shape (output_dim)
        '''
        x = x.ravel()
        y = np.dot(W, x) + b
        return y

