import numpy as np

def circShift(A, shift):
        """Shift a matrix circularly

        Args:
            A - HxW matrix
            shift - tuple containing shift value in x,y format

        Returns:
            shift_output - Circularly shifted matrix
            
        """
        temp = np.roll(A, shift[0], axis=0)
        shift_output = np.roll(temp, shift[1], axis=1)
        return shift_output

def ssd(A, B, shift):
    """Given two matrices calculate the sum of squared difference
    
    Args: 
        a,b - HxW matrices 

    Returns:
        ssd_value - scalar value of the SSD
    """
    best_match = float("inf")
    for i in range (-shift, shift):
        for j in range(-shift, shift):
            tmp = circShift(B, (j, i))
            match = np.sum((A-tmp)**2)
            if match < best_match:
                best_match = match
                shift_tup = (j, i)
    return shift_tup 

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    
    max_shift = 30;

    ## implement using SSD in brute force way
    optimal_green_shift = ssd(red, green, max_shift)
    optimal_blue_shift = ssd(red, blue, max_shift)
    new_green = circShift(green, optimal_green_shift)
    new_blue = circShift(blue, optimal_blue_shift)
    rgb_output = np.stack((red, new_green, new_blue), axis=-1)

    return rgb_output
