import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# for preprocessing
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


device = torch.device('cuda:0')
# some constants and variables
no_epochs = 8
batch_size = 60
learning_rate = 0.005
momentum = 0.9

transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307, ), (0.3081, ))
                ])
torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
train_loader = DataLoader(torchvision.datasets.EMNIST('../data/', split="balanced", train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)

test_loader = DataLoader(torchvision.datasets.EMNIST('../data/', split="balanced", train=False, download=True, transform=transform),
                         shuffle=False)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*16, 100)
        self.fc2 = nn.Linear(100, 47)

    def forward(self, x):
        output = nn.functional.relu(self.conv1(x))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = nn.functional.relu(self.conv2(output))
        output = nn.functional.max_pool2d(output, 2, 2)
        output = output.view(-1, 4*4*16)
        output = nn.functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output

model = Net()

train_acc = []
train_loss = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
for itr in range(no_epochs):
    total_loss = 0
    total_acc = 0
    for data in train_loader:

        inputs = torch.autograd.Variable(data[0])
        targets = torch.autograd.Variable(data[1])

        y_pred = model(inputs)
        predicted_target = torch.max(y_pred, 1)[1]

        loss = nn.functional.cross_entropy(y_pred, targets)
        total_loss += loss.item()
        acc = predicted_target.eq(targets.data).sum().item()/targets.size()[0]
        total_acc += acc

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    total_acc = total_acc/len(train_loader)
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

print("Training accuracy: ", train_acc[-1])
fig1 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_acc, c='r', label="train accuracy")
plt.xlabel("Epoch number"); plt.ylabel("Accuracy")
plt.title("Accuracy vs epoch"); plt.legend()
plt.show

fig2 = plt.figure()
plt.plot(np.arange(0, no_epochs), train_loss, c='r', label="train loss")
plt.xlabel("Epoch number"); plt.ylabel("Loss")
plt.title("Loss vs epoch"); plt.legend()
plt.show()

torch.save(model.state_dict(), "q7_1_4_model.pkl")

model_params = torch.load("q7_1_4_model.pkl")
model.load_state_dict(model_params)

# Run on test data
test_loss = 0
test_acc = 0
for data in test_loader:
    inputs = torch.autograd.Variable(data[0])

    targets = torch.autograd.Variable(data[1])
    y_pred = model(inputs)
    predicted_target = torch.max(y_pred, 1)[1]

    loss = nn.functional.cross_entropy(y_pred, targets)
    test_loss += loss.item()
    acc = predicted_target.eq(targets.data).sum().item()
    test_acc += acc

test_acc /= len(test_loader)
test_loss /= len(test_loader)
print("Test accuracy: ", test_acc)
print("Test loss: ", test_loss)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    image = skimage.restoration.denoise_wavelet(image, multichannel=True)
    image = skimage.color.rgb2gray(image)

    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(5))
    label_image = skimage.measure.label(bw, connectivity=2)
    props = skimage.measure.regionprops(label_image)
    mean_area = sum([x.area for x in props])/len(props)

    # Choose bounding boxes which are close in size to mean area to remove spurious dots
    bboxes = [x.bbox for x in props if x.area > mean_area*0.5]

    bw = bw.astype(np.float)
    return bboxes, bw

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
                crop_char = np.pad(crop_char, ((10, 10), (pad, pad)), 'constant', constant_values=(0, 0))
            else:
                crop_char = np.pad(crop_char, ((-pad, -pad), (10, 10)), 'constant', constant_values=(0, 0))

            crop_char = skimage.transform.resize(crop_char, (28, 28))
            crop_char = skimage.morphology.dilation(crop_char)
            crop_char = crop_char.T
            # plt.imshow(crop_char, cmap='gray')
            # plt.show()
            # flattened_char = crop_char.flatten()
            line_data.append(crop_char)
        line_data = np.asarray(line_data)
        test_data.append(line_data)


    # load the weights
    # run the crops through your neural network and print them out
    import string
    letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]]
            + ['a'] + ['b'] + ['d'] + ['e'] + ['f'] + ['g'] + ['h'] + ['n'] + ['q'] + ['r'] + ['t'])


    for line_data in test_data:
        line_data = torch.from_numpy(line_data).type(torch.float32).unsqueeze(1)

        y_pred = model(line_data)
        predicted_target = torch.max(y_pred, 1)[1]
        line_string = ''
        for pred in predicted_target.numpy():
            line_string += letters[pred]
        print(line_string)

