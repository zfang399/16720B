import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import helper
import submission as sub

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
    # F = sub.eightpoint(pts1, pts2, M)
    # print("Fundamental Matrix: \n {} \n".format(F))
    # helper.displayEpipolarF(im1, im2, F)
    # np.savez('q2_1.npz', F=F, M=M)

    # Q2.2
    # chosen_points = [24, 54, 77, 73, 75, 66, 40]
    # # chosen_points = np.random.randint(0, no_points, 7)
    # print(chosen_points)
    # pts1_seven = pts1[chosen_points]; pts2_seven = pts2[chosen_points]
    # F_list = sub.sevenpoint(pts1_seven, pts2_seven, M)
    # for F in F_list:
    #     helper.displayEpipolarF(im1, im2, F)
    # np.savez('q2_2.npz', F=F_list[0], M=M, pts1=pts1_seven, pts2=pts2_seven)

    # Q3.3
    # Run findM2.py

    # Q4.1
    # helper.epipolarMatchGUI(im1, im2, F)

    # Q5.1
    # F, inliers = sub.ransacF(pts1, pts2, M)
    # helper.displayEpipolarF(im1, im2, F)

    # Q5.3
    noisy_corresp = np.load('../data/some_corresp_noisy.npz')
    noisy_pts1 = noisy_corresp['pts1']
    noisy_pts2 = noisy_corresp['pts2']

    # F_eightpoint = sub.eightpoint(noisy_pts1, noisy_pts2, M)
    # helper.displayEpipolarF(im1, im2, F_eightpoint)

    F_sevenpoint, inliers = sub.ransacF(noisy_pts1, noisy_pts2, M)
    helper.displayEpipolarF(im1, im2, F_sevenpoint)

    E = sub.essentialMatrix(F_sevenpoint, K1, K2)
    M2_list = helper.camera2(E)

    M1 = np.eye(3)
    C1 = K1 @ np.hstack((M1, np.zeros((3, 1))))
    M2_list = helper.camera2(E)

    P = None
    M2 = None
    for i in range(M2_list.shape[-1]):
        M2_inst = M2_list[:, :, i]
        C2_inst = K1 @ M2_inst

        W_inst, err = sub.triangulate(C1, noisy_pts1[inliers], C2_inst, noisy_pts2[inliers])
        print(W_inst.shape)
        print(err)
        if np.min(W_inst[:, -1]) > 0:
            P = W_inst
            M2 = M2_inst

    C2 = K2 @ M2
    print(P.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(P[:, 0], P[:, 1], P[:, 2])

    M2, W_opt = sub.bundleAdjustment(K1, M1, noisy_pts1, K2, M2, noisy_pts2, P)

    fig = plt.figure()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(W_opt[:, 0], W_opt[:, 1], W_opt[:, 2])
    plt.show()

if __name__ == "__main__":
    main()
