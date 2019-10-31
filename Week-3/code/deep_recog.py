import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy

def test_deep_features(vgg16):
        train_data = np.load("../data/train_data.npz")

        device = torch.device('cuda')
        arglist = []

        vgg16_mod = vgg16
        vgg16_mod_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg16_mod.classifier = vgg16_mod_classifier
        for param in vgg16_mod.parameters():
            param.requires_grad = False

        arglist.append((0, "aquarium/sun_aztvjgubyrgvirup.jpg", vgg16_mod))

        features = []
        for arg in arglist:
            features.append(get_image_feature(arg))

        features = (torch.stack(features)).squeeze()
        features = features.numpy()

        image = skimage.io.imread("../data/" + "aquarium/sun_aztvjgubyrgvirup.jpg")
        my_features = network_layers.extract_deep_feature(image, util.get_VGG16_weights())

        print(features.shape, my_features.shape)
        print(scipy.spatial.distance.euclidean(features, my_features))

def build_recognition_system(vgg16, num_workers=2):
        '''
        Creates a trained recognition system by generating training features from all training images.

        [input]
        * vgg16: prebuilt VGG-16 network.
        * num_workers: number of workers to process in parallel

        [saved]
        * features: numpy.ndarray of shape (N, K)
        * labels: numpy.ndarray of shape (N)
        '''

        train_data = np.load("../data/train_data.npz")

        # ----- TODO -----
        device = torch.device('cuda')
        arglist = []

        vgg16_mod = vgg16
        vgg16_mod_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg16_mod.classifier = vgg16_mod_classifier

        for param in vgg16_mod.parameters():
            param.requires_grad = False

        for idx, filename in enumerate(train_data['files']):
            arglist.append((idx, filename, vgg16_mod))

        features = []
        for arg in arglist:
            features.append(get_image_feature(arg))

        features = (torch.stack(features)).squeeze()
        features = features.numpy()

        print(features.shape)

        np.savez("trained_system_deep.npz", features= features, labels=train_data['labels'])



def evaluate_recognition_system(vgg16, num_workers=2):
        '''
        Evaluates the recognition system for all test images and returns the confusion matrix.

        [input]
        * vgg16: prebuilt VGG-16 network.
        * num_workers: number of workers to process in parallel

        [output]
        * conf: numpy.ndarray of shape (8, 8)
        * accuracy: accuracy of the evaluated system
        '''

        test_data = np.load("../data/test_data.npz")

        # ----- TODO -----
        trained_system = np.load("trained_system_deep.npz")

        vgg16_mod = vgg16
        vgg16_mod_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg16_mod.classifier = vgg16_mod_classifier

        for param in vgg16_mod.parameters():
            param.requires_grad = False

        conf = np.zeros((8, 8))
        predicted_labels = []

        arglist = []
        for idx, filename in enumerate(test_data['files']):
            arglist.append((idx, filename, vgg16_mod))

        '''
        with multiprocessing.Pool(num_workers) as p:
            deep_test_features = p.map(get_image_feature, arglist)
        '''
        deep_test_features = []
        for arg in arglist:
            deep_test_features.append(get_image_feature(arg))

        deep_test_features = (torch.stack(deep_test_features)).squeeze()
        deep_test_features = deep_test_features.numpy()
        predicted_features =  np.argmax(distance_to_set(deep_test_features, trained_system['features']), axis = 1)

        predicted_labels = []
        for predicted_feature in predicted_features:
            predicted_label = (trained_system['labels'])[predicted_feature]
            predicted_labels.append(predicted_label)

        for idx, label in enumerate(test_data['labels']):
            conf[label, predicted_labels[idx]] += 1

        return conf

def preprocess_image(image):
        '''
        Preprocesses the image to load into the prebuilt network.

        [input]
        * image: numpy.ndarray of shape (H, W, 3)

        [output]
        * image_processed: torch.Tensor of shape (3, H, W)
        '''

        # ----- TODO -----
        # Used this to verify the network_layers CNN code
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        if len(image.shape) == 2:
            image = np.stack((image)*3, axis = -1)

        if np.max(image) > 1.0:
            image = image.astype('float')/255.0

        image_processed = skimage.transform.resize(image, (224, 224, 3))
        image_processed = (image_processed - mean)/std
        image_processed = image_processed.transpose(2, 0, 1)
        image_processed = torch.from_numpy(image_processed)

        # Commented as the outputs using pytorch transforms were not matching with network_layers outputs
        '''
        transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    ])

        image_processed = transform(image)
        '''
        return image_processed


def get_image_feature(args):
        '''
        Extracts deep features from the prebuilt VGG-16 network.
        This is a function run by a subprocess.
        [input]
        * i: index of training image
        * image_path: path of image file
        * vgg16: prebuilt VGG-16 network.

        [output]
        * feat: evaluated deep feature
        '''

        i, image_path, vgg16 = args

        # ----- TODO -----

        image = skimage.io.imread("../data/" + image_path)
        processed_image = preprocess_image(image)
        processed_image = torch.unsqueeze(processed_image, 0)
        feat = vgg16(processed_image.double())
        return feat


def distance_to_set(feature, train_features):
        '''
        Compute distance between a deep feature with all training image deep features.

        [input]
        * feature: numpy.ndarray of shape (K)
        * train_features: numpy.ndarray of shape (N, K)

        [output]
        * dist: numpy.ndarray of shape (N)
        '''

        # ----- TODO -----
        return -scipy.spatial.distance.cdist(feature, train_features, metric='euclidean')
