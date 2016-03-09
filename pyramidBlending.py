# ASSIGNMENT 6
# Alex Hagiopol 902438217

import numpy as np
import scipy as sp
import scipy.signal
import cv2
import math

""" Assignment 6 - Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def generatingKernel(parameter):
    """ Return a 5x5 generating kernel based on an input parameter.
    Args:
    parameter (float): Range of value: [0, 1].
    Returns:
    numpy.ndarray: A 5x5 kernel.
    """
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter, 0.25, 0.25 - parameter /2.0])
    return np.outer(kernel, kernel)

def reduce(image):
    """ Convolve the input image with a generating kernel of parameter of 0.4 and
    then reduce its width and height by two.
    Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)
    Returns:
    output (numpy.ndarray): an image of shape (ceil(r/2), ceil(c/2))
      For instance, if the input is 5x7, the output will be 3x4.
    """
    kernel = generatingKernel(0.4)
    convolved = sp.signal.convolve2d(image,kernel,'same')
    reduced = convolved[::2,::2]
    return reduced

def expand(image):
    """ Expand the image to double the size and then convolve it with a generating
    kernel with a parameter of 0.4.
    Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)

    Returns:
    output (numpy.ndarray): an image of shape (2*r, 2*c)
    """
    kernel = generatingKernel(0.4)
    expanded = np.zeros((image.shape[0]*2,image.shape[1]*2))
    expanded[::2,::2] = image
    convolved = sp.signal.convolve2d(expanded,kernel,'same')
    convolved = convolved * 4
    return convolved

def gaussPyramid(image, levels):
    """ Construct a pyramid from the image by reducing it by the number of levels
    passed in by the input.

    Note: You need to use your reduce function in this function to generate the
    output.
    Args:
    image (numpy.ndarray): A grayscale image of dimension (r,c) and dtype float.
    levels (uint8): A positive integer that specifies the number of reductions.

    Returns:
    output (list): A list of arrays of dtype np.float. The first element of the
                   list (output[0]) is layer 0 of the pyramid (the image
                   itself). output[1] is layer 1 of the pyramid (image reduced
                   once), etc.
    """
    output = [image]
    for i in range(1,levels + 1):
        output.append(reduce(output[i - 1]))
    return output

def laplPyramid(gaussPyr):
    """ Construct a Laplacian pyramid from the Gaussian pyramid, of height levels.
    Args:
    gaussPyr (list): A Gaussian Pyramid

    Returns:
    output (list): A Laplacian pyramid of the same size as gaussPyr.
           output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])
    """
    output = []
    for i in range(0,len(gaussPyr) - 1):
        gauss_current = gaussPyr[i]
        gauss_next = gaussPyr[i+1]
        rows = gauss_current.shape[0]
        cols = gauss_current.shape[1]
        expanded = expand(gauss_next)[:rows,:cols]
        output.append(gauss_current - expanded)
    output.append(gaussPyr[len(gaussPyr) - 1])
    return output

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """ Blend the two Laplacian pyramids by weighting them according to the
    Gaussian mask.
    Args:
    laplPyrWhite (list): A Laplacian pyramid of one image, as constructed by
                         your laplPyramid function.

    laplPyrBlack (list): A Laplacian pyramid of another image, as constructed by
                         your laplPyramid function.

    gaussPyrMask (list): A Gaussian pyramid of the mask. Each value is in the
                         range of [0, 1].
    """

    blended_pyr = []
    for i in range(0,len(laplPyrWhite)):
        whiteImg = laplPyrWhite[i]
        whiteMask = gaussPyrMask[i]
        blackImg = laplPyrBlack[i]
        blackMask = 1 - gaussPyrMask[i]
        outputImg = whiteImg*whiteMask + blackImg*blackMask
        blended_pyr.append(outputImg)
    return blended_pyr

def collapse(pyramid):
    """ Collapse an input pyramid.
    Args:
    pyramid (list): A list of numpy.ndarray images. You can assume the input is
                  taken from blend() or laplPyramid().

    Returns:
    output(numpy.ndarray): An image of the same shape as the base layer of the
                           pyramid and dtype float.
    """
    myPyramid = pyramid
    for i in reversed(range(1,len(myPyramid))):
        current_image = myPyramid[i]
        next_image = myPyramid[i-1]
        rows = next_image.shape[0]
        cols = next_image.shape[1]
        expanded = expand(current_image)[:rows,:cols]
        myPyramid[i-1] = next_image + expanded
    return myPyramid[0]

def pyramidBlend(black_image, white_image, mask):
    """ This function blends the two images according to
    mask.

    Assume all images are float dtype, and return a float dtype.
    """
    # Automatically figure out the size
    min_size = min(black_image.shape)
    depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

    gauss_pyr_mask = gaussPyramid(mask, depth)
    gauss_pyr_black = gaussPyramid(black_image, depth)
    gauss_pyr_white = gaussPyramid(white_image, depth)


    lapl_pyr_black  = laplPyramid(gauss_pyr_black)
    lapl_pyr_white = laplPyramid(gauss_pyr_white)

    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    outimg = collapse(outpyr)

    outimg[outimg < 0] = 0 # blending sometimes results in slightly out of bound numbers.
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)
    return outimg