import numpy as np
import cv2
import BRIEF
import matplotlib.pyplot as plt

if __name__ == '__main__':
    im = cv2.imread("../data/model_chickenbroth.jpg")
    imH, imW, _ = im.shape
    locs, desc = BRIEF.briefLite(im)
    fig = plt.figure()
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    #plt.plot(locs[:,0], locs[:,1], 'r.')
    #plt.draw()
    #plt.waitforbuttonpress(0)
    #plt.close(fig)

    no_matches = []
    for i in range(0, 360, 10):
        rotation_matrix = cv2.getRotationMatrix2D((imW//2, imH//2), i, 1)
        im_rot = cv2.warpAffine(im, rotation_matrix, (imH, imH))
        locs_rot, desc_rot = BRIEF.briefLite(im_rot)
        matches = BRIEF.briefMatch(desc, desc_rot)
        no_matches.append(len(matches))
        #BRIEF.plotMatches(im, im_rot, matches, locs, locs_rot)

    fig1 = plt.figure()
    plt.bar(np.arange(36), no_matches)
    plt.xlabel("Rotation angle/10")
    plt.ylabel("No of matches between images")
    #plt.show()
