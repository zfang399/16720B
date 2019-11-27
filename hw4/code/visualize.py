'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import cv2
import helper
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    """main function
    :returns: TODO

    """
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    M = max(im1.shape)
    corresp_points = np.load('../data/some_corresp.npz')
    pts1 = corresp_points['pts1']; pts2 = corresp_points['pts2']
    F = sub.eightpoint(pts1, pts2, M)

    templeCoords = np.load('../data/templeCoords.npz')
    x1 = templeCoords['x1']; y1 = templeCoords['y1']

    x2 = np.zeros((x1.shape[0])); y2 = np.zeros((x1.shape[0]))

    for i in range(x1.shape[0]):
        x2[i], y2[i] = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])

    temple_pts1 = np.hstack((x1, y1))
    temple_pts2 = np.hstack((x2[:, None], y2[:, None]))

    plt.scatter(temple_pts1[:, 0], temple_pts1[:, 1])
    plt.show()

    plt.scatter(temple_pts2[:, 0], temple_pts2[:, 1])
    plt.show()

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']; K2 = intrinsics['K2']
    E = sub.essentialMatrix(F, K1, K2)

    M1 = np.eye(3)
    C1 = K1 @ np.hstack((M1, np.zeros((3, 1))))
    M2_list = helper.camera2(E)

    P = []
    M2 = []
    for i in range(M2_list.shape[-1]):
        M2_inst = M2_list[:, :, i]
        C2_inst = K1 @ M2_inst

        W_inst, err = sub.triangulate(C1, temple_pts1, C2_inst, temple_pts2)
        print(W_inst.shape)
        print(err)
        if np.min(W_inst[:, -1]) > 0:
            P = W_inst
            M2 = M2_inst

    C2 = K2 @ M2
    print(P.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
