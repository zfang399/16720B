import numpy as np
import cv2
import os
from planarH import computeH
import matplotlib.pyplot as plt

def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    K_inv = np.linalg.inv(K)
    H_new = np.matmul(K_inv, H)
    U, s, Vh = np.linalg.svd(H_new[:, 0:2], full_matrices=True)
    # Rotation matrix should not cause any shearing in the dimensions
    R = np.matmul(U, np.vstack((np.eye(2), [0, 0])))
    R = np.matmul(R, Vh)
    col1_R = np.array(R[:, 0]); col2_R = np.array(R[:, 1]); col3_R = np.array(np.cross(R[:, 0], R[:, 1]))
    R = np.vstack((col1_R, col2_R, col3_R)).T

    # eliminate reflection component in the Rotation matrix
    if round(np.linalg.det(R)) == -1:
        R = R[:, 2] * -1

    scaling_factor = np.sum(H_new[:, 0:2] / R[:, 0:2])/6
    t = H_new[:, 2]/scaling_factor

    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...
    # augment R and t
    R_t = np.vstack((R[:, 0], R[:, 1], R[:, 2], t)).T

    # pre-multiply K with R_t
    K_R_t = np.matmul(K, R_t)

    # make the 3D points homogeneous
    W = np.vstack((W[0, :], W[1, :], W[2, :], np.ones(W.shape[1])))

    # X should be homogeneous 2D projected coordinates
    X = np.matmul(K_R_t, W)
    X = X / np.array([X[2, :]])
    X = X[0:2, :]
    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')
    #############################
    # TO DO ...
    # perform required operations and plot sphere
    W = np.loadtxt('../data/sphere.txt')

    # given input data
    K = np.array([[3043.72, 0, 1196.0], [0, 3043.72, 1604.00], [0, 0, 1]])
    U = np.array([[0, 0, 0], [18.2, 0, 0], [18.2, 26.0, 0.0], [0, 26.0, 0]]).T
    X = np.array([[483, 810], [1704, 781], [2175, 2217], [67, 2286]]).T


    # compute the homography for the given 3D points (book corners) to 2D coords in image
    H = computeH(X, U[0:2])

    # translate the 3D points of the sphere to align to the 'o' on the book
    o_coord_2d = [830, 1645, 1] # by inspection
    o_coord_3d = np.matmul(np.linalg.inv(H), o_coord_2d)
    o_coord_3d = o_coord_3d/o_coord_3d[2]
    o_coord_3d[2] = 6.8581/2 # diameter of sphere is 6.8581 given
    W = W + o_coord_3d[:, None]

    # estimate planar R, t
    R, t = compute_extrinsics(K, H)
    X = project_extrinsics(K, W, R, t)
    X = X.astype(int)


    # translate the projected points to align to the 'o' on the book
    #X[0, :] += 320
    #X[1, :] += 640

    # plot the points and the image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.plot(X[0, :], X[1, :], c='y', linewidth=0.5)
    plt.show()
