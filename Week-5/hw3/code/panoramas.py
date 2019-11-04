import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix.
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''

    #######################################
    # TO DO ...
    imH1, imW1, _ = im1.shape
    imH2, imW2, _ = im2.shape
    width = round(max(imW1, imW2)*1.5)
    warped_im2 = cv2.warpPerspective(im2, H2to1, (width, im2.shape[0]))
    cv2.waitKey(0)

    # pad image 1 to make it the same size of im2
    im1 = cv2.copyMakeBorder(im1, 0, imH2 - imH1, 0, width - imW1, cv2.BORDER_CONSTANT, 0)
    pano_im = np.maximum(im1, warped_im2)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping.
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''

    ######################################
    # TO DO ...
    imH1, imW1, _ = im1.shape
    imH2, imW2, _ = im2.shape
    width = round(max(imW1, imW2)*1.25)
    new_corners = np.dot(H2to1, np.array([[0, 0, 1],
                                          [imW2, 0, 1],
                                          [0, imH2, 1],
                                          [imW2, imH2, 1]]).T).T
    # normalize the homogeneous corner coords
    new_corners = np.round(new_corners / np.array([new_corners[:, -1]]).T)

    min_x = min(new_corners[0, 0], 0, new_corners[2, 0])
    max_x = max(new_corners[1, 0], imW1, new_corners[3, 0])
    min_y = min(new_corners[2, 1], 0, new_corners[1, 1])
    max_y = max(new_corners[3, 1], imH1, new_corners[2, 1])

    ratio = width/(max_x - min_x)
    height = int(round(ratio*(max_y - min_y)))

    M = np.zeros((3, 3))
    M[2, 2] = 1
    M[0, 0] = ratio
    M[1, 1] = ratio
    M[0, 2] = -min_x*ratio
    M[1, 2] = -min_y*ratio
    out_size = (width, height)

    warped_im1 = cv2.warpPerspective(im1, M, out_size)
    warped_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), out_size)

    # Blend the two images to create a panorama
    mask = np.ones((im1.shape))
    mask[1:-1, 1:-1, :] = 0
    mask = distance_transform_edt(1-mask)
    mask = mask/np.amax(mask)
    warp_mask_im1 = cv2.warpPerspective(mask, M, out_size)

    mask = np.ones((im2.shape))
    mask[1:-1, 1:-1, :] = 0
    mask = distance_transform_edt(1-mask)
    mask = mask/np.amax(mask)
    warp_mask_im2 = cv2.warpPerspective(mask, np.matmul(M, H2to1), out_size)

    mask_1 = np.divide(warp_mask_im1, warp_mask_im1 + warp_mask_im2)
    mask_2 = np.divide(warp_mask_im2, warp_mask_im1 + warp_mask_im2)
    mask_1 = np.where(mask_1 == np.nan, 0, mask_1)
    mask_2 = np.where(mask_2 == np.nan, 0, mask_2)

    pano_im = np.uint8(np.multiply(warped_im1, mask_1) + np.multiply(warped_im2, mask_2))
    return pano_im

def generatePanorama(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping)
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)
    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imshow("Panorama", pano_im)
    cv2.waitKey(0)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    generatePanorama(im1, im2)
