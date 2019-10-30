import numpy as np
import cv2
import submission as sub
import helper
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics, M):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :param M:
    :return: M2, the extrinsics of camera 2
                     C2, the 3x4 camera matrix
                     P, 3D points after triangulation (Nx3)
    '''
    K1 = intrinsics['K1']; K2 = intrinsics['K2']
    F = sub.eightpoint(pts1, pts2, M)
    E = sub.essentialMatrix(F, K1, K2)

    M1 = np.eye(3)
    C1 = K1 @ np.hstack((M1, np.zeros((3, 1))))
    M2_list = helper.camera2(E)

    P = []
    M2 = []
    for i in range(M2_list.shape[-1]):
        M2_inst = M2_list[:, :, i]
        C2_inst = K1 @ M2_inst

        W_inst, err = sub.triangulate(C1, pts1, C2_inst, pts2)
        print(err)
        if np.min(W_inst[:, -1]) > 0:
            P = W_inst
            M2 = M2_inst

    C2 = K2 @ M2
    return M2, C2, P


if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    intrinsics = np.load('../data/intrinsics.npz')
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    M = max(im1.shape)

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics, M)
    np.savez('q3_3', M2=M2, C2=C2, P=P)
