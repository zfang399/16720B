import matplotlib.pyplot as plt
import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    layer_num = 3
    arglist = []
    for idx, filename in enumerate(train_data['files']):
        arglist.append(("../data/" + filename, dictionary, layer_num, dictionary.shape[0]))

    with multiprocessing.Pool(num_workers) as p:
        features = p.starmap(get_image_feature, arglist)

    features = np.array(features)
    print("Feature vector shape = {}".format(features.shape))
    np.savez('trained_system.npz', features = features, labels = train_data['labels'], dictionary=dictionary, layer_num = layer_num)
    print("Trained system saved as trained_system.npz")

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    conf = np.zeros((8, 8))
    predicted_labels = []
    for filename in test_data['files']:
        test_image = skimage.io.imread("../data/" + filename)
        test_wordmap = visual_words.get_visual_words(test_image, trained_system['dictionary'])
        test_features = get_feature_from_wordmap_SPM(test_wordmap, trained_system['layer_num'], trained_system['dictionary'].shape[0])
        predicted_feature = np.argmax(distance_to_set(test_features, trained_system['features']))
        predicted_label = (trained_system['labels'])[predicted_feature]
        print("Predicted Label: {}".format(predicted_label))
        predicted_labels.append(predicted_label)

    for idx, label in enumerate(test_data['labels']):
        conf[label, predicted_labels[idx]] += 1
        if label != predicted_labels[idx]:
            print(test_data['files'][idx], label, predicted_labels[idx])

    accuracy = conf.trace()/np.sum(conf)
    return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    image = skimage.io.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return features

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    minima = np.minimum(word_hist, histograms)
    intersection = np.sum(minima, axis=1)
    return intersection

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    hist = np.histogram(wordmap, dict_size)
    hist = np.array(hist[0])

    # normalize the histogram
    hist = hist/np.sum(hist)
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):

    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    h, w = wordmap.shape
    hist_all = []
    # TODO: Optimize the loop to reuse the histogram
    for layer in range(layer_num-1, -1, -1):
        tile_height = math.floor(h/(2**layer))
        tile_width  = math.floor(w/(2**layer))
        x, y = 0, 0
        histo = []
        for i in range(2**layer, 0, -1):
            x = 0
            for j in range(2**layer, 0, -1):
                hist_patch = get_feature_from_wordmap(wordmap[x:x+tile_height, y:y+tile_width], dict_size)

                if layer <= 1:
                    weight = float(2)**(-layer_num)
                else:
                    weight = float(2)**(layer - layer_num - 1)

                histo = np.append(histo, weight*hist_patch)

                x = x + tile_height
            y = y + tile_width

        hist_level = np.array(histo)
        hist_level = np.reshape(hist_level, (dict_size, 2**layer, 2**layer))
        # normalize histogram
        hist_level = hist_level / np.sum(hist_level)
        hist_all = np.append(hist_all, hist_level)

    hist_all = np.array(hist_all)
    hist_all = hist_all / np.sum(hist_all)
    return hist_all

