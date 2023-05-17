import imageio
import skimage
from scipy.signal import convolve2d
import numpy as np
import scipy.ndimage.filters as sc
from skimage.color import rgb2gray
COLOR_DIMS = 3
GRAYSCALE = 1
MAX_Z = 255

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imageio.imread(filename)
    if image.ndim == COLOR_DIMS and representation == GRAYSCALE:
        return rgb2gray(image)
    return image.astype(np.float64) / MAX_Z

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def create_filter(filter_size):
    base_filter = np.asarray([1, 1])
    filter = np.asarray([1, 1])
    while filter.size != filter_size:  # todo while or for!!
        filter = np.convolve(filter, base_filter)
    filter = filter.reshape(1, filter_size)
    return filter / np.sum(filter)

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    res_vec = create_filter(filter_size)
    pyr = []
    pyr.append(im)
    for i in range(1, max_levels):

        im = reduce(im, res_vec)
        if im.shape[0] < 16 or im.shape[1] < 16:
            break
        pyr.append(im)
    return pyr, res_vec

def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    image_copy = np.copy(im)
    image_convolve_row = sc.convolve(image_copy, blur_filter)
    image_after_convolve = sc.convolve(image_convolve_row, blur_filter.T)
    new_image = image_after_convolve[::2, ::2]
    return new_image


