import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''

    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    for level in levels[1:]:
        DoG_pyramid.append(gaussian_pyramid[..., level+1] - gaussian_pyramid[..., level])
    DoG_pyramid = np.asarray(DoG_pyramid)
    DoG_pyramid = DoG_pyramid.transpose(1, 2, 0)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None

    ##################
    # TO DO ...
    # Compute principal curvature here
    D_xx = cv2.Sobel(DoG_pyramid, -1, dx=2, dy=0, ksize=3)
    D_yy = cv2.Sobel(DoG_pyramid, -1, dx=0, dy=2, ksize=3)
    D_xy = cv2.Sobel(DoG_pyramid, -1, dx=1, dy=1, ksize=3)

    principal_curvature = (D_xx + D_yy)**2/(D_xx * D_yy - (D_xy)**2)

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None

    ##############
    #  TO DO ...
    # Compute locsDoG here
    X = []; Y = []; Z = []
    for level in DoG_levels:
        stacked = np.dstack((DoG_pyramid[1:-1, 1:-1, level],
                             DoG_pyramid[1:-1, :-2, level],
                             DoG_pyramid[:-2, 1:-1, level],
                             DoG_pyramid[1:-1, 2:, level],
                             DoG_pyramid[2:, 1:-1, level],
                             DoG_pyramid[:-2, 2:, level],
                             DoG_pyramid[2:, :-2, level],
                             DoG_pyramid[2:, 2:, level],
                             DoG_pyramid[:-2, :-2, level]))
        # if first level only add upper layer neighbor comparison
        if level == DoG_levels[0]:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level+1]))
        # if last level only add the lower layer neighbor comparison
        elif level == DoG_levels[-1]:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level-1]))
        else:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level-1],
                                          DoG_pyramid[1:-1, 1:-1, level+1]))

        # find max and min
        max_stacked = np.amax(stacked, axis = 2)
        min_stacked = np.amin(stacked, axis = 2)
        # if pixel in question is min or max
        keypoint_matrix_max = np.where(DoG_pyramid[1:-1, 1:-1, level] == max_stacked,
                                        DoG_pyramid[1:-1, 1:-1, level], 0)
        keypoint_matrix_min = np.where(DoG_pyramid[1:-1, 1:-1, level] == min_stacked,
                                        DoG_pyramid[1:-1, 1:-1, level], 0)
        # as max and min of the same elements are mutually exclusive unless all elements are same
        keypoint_matrix = keypoint_matrix_max + keypoint_matrix_min

        # vet the points if they don't conform to the thresholds
        keypoint_matrix = np.where(keypoint_matrix > th_contrast, keypoint_matrix, 0)
        keypoint_matrix = np.where(principal_curvature[1:-1, 1:-1, level] < th_r, keypoint_matrix, 0)

        # get coordinates of all non-zero elements
        x, y = np.nonzero(keypoint_matrix)
        z = np.full(len(x), level)
        X.extend(x+1); Y.extend(y+1); Z.extend(z)

    locsDoG = np.stack([np.asarray(Y), np.asarray(X), np.asarray(Z)], axis = -1)
    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''

    ##########################
    # TO DO ....
    # compute gauss_pyramid, locsDoG here
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im =cv2.imread('../data/pf_scan_scaled.jpg')
    #im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)

    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)

    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # plot the keypoints as circles
    for keypoint in locsDoG:
        c = (int(keypoint[0]), int(keypoint[1]))
        cv2.circle(im, c,1, [0, 255, 0], 1)
    #cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
    #cv2.imshow("Keypoints",im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
