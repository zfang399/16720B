import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                equation
    '''

    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    p2_hmg = np.stack((p2[0], p2[1], np.ones(p2.shape[1])), axis = 1)
    x_p2_hmg = p2_hmg*(np.transpose([-p1[0]]))
    y_p2_hmg = p2_hmg*(np.transpose([p1[1]]))

    first = np.empty((p2_hmg.shape[0]*2, p2_hmg.shape[1]))
    first[0::2] = 0
    first[1::2] = p2_hmg

    second = np.empty((p2_hmg.shape[0]*2, p2_hmg.shape[1]))
    second[0::2] = -p2_hmg
    second[1::2] = 0

    third = np.empty((p2_hmg.shape[0]*2, p2_hmg.shape[1]))
    third[0::2] = y_p2_hmg
    third[1::2] = x_p2_hmg

    a_matrix = np.hstack((first, second, third))

    u, s, vh = np.linalg.svd(a_matrix)
    H2to1 = np.reshape(vh[-1, :], (3, 3))
    H2to1 = H2to1/np.linalg.norm(H2to1)
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC

    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    # TO DO ...
    p = np.transpose(locs1)[0:2, ...]; q = np.transpose(locs2)[0:2, ...]
    inliers = []
    H_list = []
    for iteration in range(0, num_iter):
        # randomly sample minimal (4) points and compute H
        sampled_matches = matches[np.random.randint(matches.shape[0], size=4), :]
        random_matches_p = p[:, sampled_matches[:, 0]]; random_matches_q = q[:, sampled_matches[:, 1]]
        H = computeH(random_matches_p, random_matches_q)

        # transform q by H and check number of inliers
        p_matches = p[:, matches[:, 0]]
        q_matches = q[:, matches[:, 1]]
        p_hmg = np.stack((p_matches[0], p_matches[1], np.ones(p_matches.shape[1])), axis = 1)
        q_hmg = np.stack((q_matches[0], q_matches[1], np.ones(q_matches.shape[1])), axis = 1)
        trans_q = np.dot(H, q_hmg.T).T
        trans_q = trans_q / np.transpose([trans_q[:, -1]])

        # compute inliers for the transform
        inlier_distances = np.linalg.norm(p_hmg - trans_q, axis=1)

        # remove all points farther than the tolerance distance
        inlier = np.where(inlier_distances <= tol, 1, 0)
        inliers.append(np.sum(inlier))
        H_list.append(H)

    # best H is the one with maximum inliers
    idx = np.argmax(np.asarray(inliers))
    bestH = H_list[idx]
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
