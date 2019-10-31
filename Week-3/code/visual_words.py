import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import glob

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----

    no_of_filters = 20

    # sanity checks on the image
    image = skimage.img_as_float(image)
    if len(image.shape) < 3:
        image = np.dstack((image, image, image))
    image = skimage.color.rgb2lab(image)

    # filter the images
    scales = [1, 2, 4, 8, 8*np.sqrt(2)]
    filter_resp = np.empty([image.shape[0], image.shape[1], 0])
    for scale in scales:
        filter_gauss    =  scipy.ndimage.gaussian_filter(image, sigma=(scale, scale, 0))

        filter_laplace = np.empty([image.shape[0], image.shape[1], 3])
        filter_laplace[:, :, 0]  =  scipy.ndimage.gaussian_laplace(image[:, :, 0], sigma=scale)
        filter_laplace[:, :, 1]  =  scipy.ndimage.gaussian_laplace(image[:, :, 1], sigma=scale)
        filter_laplace[:, :, 2]  =  scipy.ndimage.gaussian_laplace(image[:, :, 2], sigma=scale)
        filter_gauss_x  =  scipy.ndimage.gaussian_filter(image, sigma=(scale, scale, 0), order=(0, 1, 0))
        filter_gauss_y  =  scipy.ndimage.gaussian_filter(image, sigma=(scale, scale, 0), order=(1, 0, 0))

        filter_resp     = np.append(filter_resp, np.concatenate([filter_gauss,
                                                                 filter_laplace,
                                                                 filter_gauss_x,
                                                                 filter_gauss_y], axis=2), axis=2)
    return filter_resp

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    filter_resp = extract_filter_responses(image)
    filter_resp = filter_resp.reshape(filter_resp.shape[0]*filter_resp.shape[1], filter_resp.shape[2])

    distances = scipy.spatial.distance.cdist(filter_resp, dictionary)
    distances = distances.reshape(image.shape[0], image.shape[1], distances.shape[1])
    wordmap = np.argmin(distances, axis=2)
    wordmap = wordmap.astype('float')/255
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''
    i, alpha, image_path = args
    # ----- TODO -----
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255

    filter_resp = extract_filter_responses(image)
    rand_pixels = np.random.permutation(alpha)
    filter_resp = filter_resp.reshape(filter_resp.shape[0]*filter_resp.shape[1], filter_resp.shape[2])
    filter_resp = filter_resp[rand_pixels, :]
    np.save('outfile_'+ str(i), filter_resp)

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    alpha = 400

    arglist = []
    for idx, filename in enumerate(train_data['files']):
        arglist.append((idx, alpha, '../data/' + filename))

    with multiprocessing.Pool(num_workers) as p:
        p.map(compute_dictionary_one_image, arglist)

    filter_files = glob.glob("./outfile*.npy")
    filter_resp_array = np.vstack([np.load(f) for f in filter_files])
    print(filter_resp_array.shape)

    kmeans = sklearn.cluster.KMeans(n_clusters=200).fit(filter_resp_array)
    dictionary = kmeans.cluster_centers_
    print(dictionary.shape)
    np.save('dictionary', dictionary)
