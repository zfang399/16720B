import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    mean_height = sum([bbox[2] - bbox[0] for bbox in bboxes])/len(bboxes)
    # Center: coordx, coordy, width, height
    centers = [[(bbox[3]+bbox[1])//2, (bbox[2]+bbox[0])//2, bbox[3]-bbox[1], bbox[2]-bbox[0]] for bbox in bboxes]
    # Sort the centers according to coordy (top to bottom)
    centers.sort(key = lambda center: center[1])

    rows = []
    current_row_y = centers[0][1]
    row = []
    for c in centers:
        # Sort according to coordx(left to right)
        if c[1] > current_row_y + mean_height:
            row = sorted(row, key=lambda c: c[0])
            rows.append(row)
            row = [c]
            current_row_y = c[1]
        else:
            row.append(c)
    # last row is not appended in rows
    row = sorted(row, key=lambda c:c[0])
    rows.append(row)





    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    test_data = []
    for row in rows:
        line_data = []
        for x, y, width, height in row:
            crop_char = bw[y-height//2:y+height//2, x-width//2:x+width//2]
            # pad the cropped character to square size
            pad = (height-width)//2 + 10
            if pad > 0:
                crop_char = np.pad(crop_char, ((10, 10), (pad, pad)), 'constant', constant_values=(1, 1))
            else:
                crop_char = np.pad(crop_char, ((-pad, -pad), (10, 10)), 'constant', constant_values=(1, 1))

            crop_char = skimage.transform.resize(crop_char, (32, 32))
            # since it is a negative image
            crop_char = skimage.morphology.erosion(crop_char)
            crop_char = crop_char.T
            flattened_char = crop_char.flatten()
            line_data.append(flattened_char)
        line_data = np.asarray(line_data)
        test_data.append(line_data)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    for line_data in test_data:
        h1 = forward(line_data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        pred_idx = np.argmax(probs, axis = -1)
        line_string = ''
        for pred in pred_idx:
            line_string += letters[pred]
        print(line_string)
