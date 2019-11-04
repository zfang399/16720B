'''
Written by Akash Sharma
October 2019
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import helper
import submission as sub
import findM2

def reprojection_error(K1, M1, pts1, K2, M2, pts2, W):
    """Calculate the reprojection error for W

    :K1: TODO
    :M1: TODO
    :pts1: TODO
    :K2: TODO
    :M2: TODO
    :pts2: TODO
    :W: TODO
    :returns: error

    """
    C1 = K1 @ M1
    C2 = K2 @ M2
    homo_W = np.hstack((W, np.ones((W.shape[0], 1))))
    pts1_proj = (C1 @ homo_W.T).T
    pts2_proj = (C2 @ homo_W.T).T
    pts1_proj = (pts1_proj/pts1_proj[:, -1][:, None])[:, 0:2]
    pts2_proj = (pts2_proj/pts2_proj[:, -1][:, None])[:, 0:2]
    err = np.sum(np.square(pts1 - pts1_proj) + np.square(pts2 - pts2_proj))
    return err


def main():
    """Main function
    :returns: TODO

    """
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    M = max(im1.shape)
    corresp_points = np.load('../data/some_corresp.npz')
    pts1 = corresp_points['pts1']; pts2 = corresp_points['pts2']
    no_points = pts1.shape[0]
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']; K2 = intrinsics['K2']


    # Q2.1
    F_eightpoint = sub.eightpoint(pts1, pts2, M)
    print("Fundamental Matrix from eightpoint: \n {} \n".format(F_eightpoint))
    helper.displayEpipolarF(im1, im2, F_eightpoint)
    np.savez('q2_1.npz', F=F_eightpoint, M=M)

    # Q2.2
    chosen_points = [45, 29, 10, 59, 5, 53, 38]
    pts1_seven = pts1[chosen_points]; pts2_seven = pts2[chosen_points]
    Farray = sub.sevenpoint(pts1_seven, pts2_seven, M)
    for F in Farray:
        helper.displayEpipolarF(im1, im2, F)
    F_sevenpoint = Farray[0]
    print("Fundamental Matrix from sevenpoint: \n {} \n".format(F_sevenpoint))
    np.savez('q2_2.npz', F=F_sevenpoint, M=M, pts1=pts1_seven, pts2=pts2_seven)

    # Q3.1
    E = sub.essentialMatrix(F_eightpoint, K1, K2)
    print("Essential matrix obtained from F_eightpoint: \n {} \n".format(E))

    # Q3.3
    # Run findM2.py

    # Q4.1
    clicked_pts1, epi_corresp_pts = helper.epipolarMatchGUI(im1, im2, F_eightpoint)
    np.savez('q4_1.npz', F=F_eightpoint, pts1=clicked_pts1, pts2=epi_corresp_pts)

    # Q4.2
    # Run visualize.py

    # Q5.1
    noisy_corresp = np.load('../data/some_corresp_noisy.npz')
    noisy_pts1 = noisy_corresp['pts1']
    noisy_pts2 = noisy_corresp['pts2']

    F_eightpoint = sub.eightpoint(noisy_pts1, noisy_pts2, M)
    helper.displayEpipolarF(im1, im2, F_eightpoint)

    F_sevenpoint, inliers = sub.ransacF(noisy_pts1, noisy_pts2, M)
    helper.displayEpipolarF(im1, im2, F_sevenpoint)

    # Q5.3
    E = sub.essentialMatrix(F_sevenpoint, K1, K2)

    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    M2, _, P = findM2.test_M2_solution(noisy_pts1[inliers, :], noisy_pts2[inliers, :], intrinsics, M)

    M2_opt, W_opt = sub.bundleAdjustment(K1, M1, noisy_pts1[inliers, :], K2, M2, noisy_pts2[inliers, :], P)

    # Compute the reprojection error
    err_before_BA = reprojection_error(K1, M1, noisy_pts1[inliers, :], K2, M2, noisy_pts2[inliers, :], P)
    err_after_BA = reprojection_error(K1, M1, noisy_pts1[inliers, :], K2, M2_opt, noisy_pts2[inliers, :], W_opt)
    print("Reprojection error before BA: {} \n Reprojection error after BA: {}".format(err_before_BA, err_after_BA))

    # Plot the 3D reconstruction of noisy points
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    z_max1 = np.amax(P[:, 2])
    ax1.scatter(P[:, 0]/z_max1, P[:, 1]/z_max1, P[:, 2]/z_max1, label='Before Bundle Adjustment')

    z_max2 = np.amax(W_opt[:, 2])
    ax1.scatter(W_opt[:, 0]/z_max2, W_opt[:, 1]/z_max2, W_opt[:, 2]/z_max2, c ='r', marker='o', label='After Bundle Adjustment')
    ax1.legend()
    plt.show()

if __name__ == "__main__":
    main()
