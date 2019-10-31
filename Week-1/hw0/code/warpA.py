import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    
    A_inv = np.linalg.inv(A)
    output_im = np.empty((output_shape[0], output_shape[1]), dtype=np.float)
    
    coords_x, coords_y = np.indices(output_shape)
    coords = np.stack((coords_x.ravel(), coords_y.ravel(), np.ones(coords_y.size)))
    coords_vec = np.reshape(coords, (3, output_shape[0]*output_shape[1]))
    
    coords_vec = np.round(A_inv.dot(coords_vec))
    coords_vec = np.ndarray.astype(coords_vec, int)
    coords_vec[0][coords_vec[0] >= output_shape[0]] = 0
    coords_vec[1][coords_vec[1] >= output_shape[1]] = 0
    
    output_im = np.where(coords_vec[0] > 0, im[coords_vec[0], coords_vec[1]], 0) 
    output_im = np.where(coords_vec[1] > 0, output_im, 0)
    output_im = output_im.reshape(output_shape[0], output_shape[1]) 
   
    return output_im
