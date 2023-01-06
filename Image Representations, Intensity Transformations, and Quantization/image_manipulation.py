import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import skimage.color

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
MAX_Z = 255
COLOR_DIMS = 3


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imageio.imread(filename)
    if image.ndim == COLOR_DIMS and representation == GRAYSCALE:
        return skimage.color.rgb2gray(image)
    return image.astype(np.float64) / MAX_Z


def imdisplay(filename, representation):
    """
        Reads an image and displays it into a given representation
        :param filename: filename of image on disk
        :param representation: 1 for greyscale and 2 for RGB
        """
    image = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """

    return np.matmul(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """

    yiq_rgb_transformation_matrix = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
    return np.matmul(imYIQ, yiq_rgb_transformation_matrix.T)


def build_channel(image):
    """
    isolate the relevent channel to work on
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        return rgb2yiq(image)[:, :, 0], True
    else:
        return image, False


def initial_z(n_quant):
    """
    initial z places
    :param n_quant:
    :return:
    """
    z_places = np.zeros(n_quant + 1, dtype=np.int64)
    z_places[-1], z_places[0] = 255, -1
    return z_places


def initial_q(z_places):
    """
    initial q_places
    :param z_places:
    :return:
    """
    q = np.zeros(z_places.size - 1)
    for i in range(0, len(z_places) - 1):
        q[i] = np.floor((z_places[i] + z_places[i + 1]) / 2)
    return q


def z_loop(z_places, num_of_pixel_in_seg, cum_hist, n_quant):
    for i in range(1, n_quant):
        z_places[i] = np.argmax(cum_hist > num_of_pixel_in_seg * i) - 1


def initial_z_q(n_quant, image_histogram):
    """
    initial z and q places
    :param n_quant:
    :param image_histogram:
    :return:
    """
    z_places = initial_z(n_quant)
    num_of_pixel_in_seg = int(sum(image_histogram) / n_quant)
    cum_hist = np.cumsum(image_histogram)
    z_loop(z_places, num_of_pixel_in_seg, cum_hist, n_quant)
    q_places = initial_q(z_places).astype(np.int64)
    return (q_places, z_places)


def z_i_calculator(index, q_places):
    """
    calculate new z according to formula
    :param index:
    :param q_places:
    :return:
    """
    return int(np.floor((q_places[index] + q_places[index - 1]) / 2))


def q_i_calculator(index, z_places, image_histogram):
    """
    calculate new q according to formula
    :param index:
    :param q_places:
    :return:
    """

    upper_sum = (np.arange(z_places[index] + 1, z_places[index + 1] + 1)) @ (
        image_histogram[z_places[index] + 1:z_places[index + 1] + 1].T)
    lower_sum = sum(image_histogram[z_places[index]+1:z_places[index + 1] + 1])
    return int(upper_sum / lower_sum) if lower_sum else 0


def error_calculator(q_places, image_histogram, z_places, n_quant):
    """
    calculate the error
    :param q_places:
    :param image_histogram:
    :param z_places:
    :param n_quant:
    :return:
    """
    sum = 0
    for i in range(n_quant):
        change = q_places[i] - np.arange(z_places[i] + 1, z_places[i + 1] + 1)
        change = change ** 2
        hist_part = image_histogram[z_places[i] + 1:z_places[i + 1] + 1].T
        sum += change @ hist_part

    return sum


def optimize(z_places, q_places, image_histogram, n_quant, n_iter):
    """
    optimization function, loops are at most constant in size
    :param z_places:
    :param q_places:
    :param image_histogram:
    :param n_quant:
    :param n_iter:
    :return:
    """

    error = []
    z_copy = z_places.copy()
    for i in range(n_iter):
        for j in range(1, n_quant):
            z_places[j] = z_i_calculator(j, q_places)
        for j in range(n_quant):
            q_places[j] = q_i_calculator(j, z_places, image_histogram)
        if np.array_equal(z_copy, z_places):
            break
        else:
            z_copy = z_places.copy()
        error.append(error_calculator(q_places, image_histogram, z_places, n_quant))

    return np.asarray(error)


def paint_channel(channel, z_places, n_quant, q_places):
    """
    create the image back after changes
    :param channel:
    :param z_places:
    :param n_quant:
    :param q_places:
    :return:
    """
    channel *= 255
    channel = channel.astype(np.int64)
    for i in range(n_quant):
        channel[(channel <= z_places[i + 1]) & (channel >= z_places[i])] = q_places[i]
    channel = channel / 255
    return channel





def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """

    channel, color = build_channel(im_orig.copy())  # build gray/y channel
    image_histogram, _ = np.histogram(channel.flatten(), bins=256, range=(0, 1))
    q_places, z_places = initial_z_q(n_quant, image_histogram)
    error = optimize(z_places, q_places, image_histogram, n_quant, n_iter)
    channel = paint_channel(channel, z_places, n_quant, q_places)  # channel now 0-1
    if not color:
        return channel, error
    else:
        color_image = rgb2yiq(im_orig.copy())
        color_image[:, :, 0] = channel
        color_image = yiq2rgb(color_image)
        return color_image, error


def return_grad():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([MAX_Z] * 6)[None, :]])
    im_orig = np.tile(x, (256, 1)) / 255
    return im_orig


def build_nch(image_histogram):
    cumulative_histogram = np.cumsum(image_histogram)
    c_m = cumulative_histogram[np.nonzero(cumulative_histogram)[0][0]]
    nch = (MAX_Z * (cumulative_histogram - c_m)) / (cumulative_histogram[255] - c_m)
    nch = np.floor(nch)
    return nch


def build_new_channel(channel, nch):
    channel = channel * 255
    channel = channel.astype(np.int64)
    channel = nch[channel]
    return channel / 255


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    image = im_orig.copy()
    channel, color = build_channel(image)
    image_histogram, _ = np.histogram(channel, bins=256, range=(0, 1))
    nch = build_nch(image_histogram)
    channel = build_new_channel(channel, nch)  # channel is 0-1
    hist_eq, _ = np.histogram(channel, bins=256, range=(0, 1))
    plt.show()
    if not color:
        return [channel, image_histogram, hist_eq]
    else:
        color_image = rgb2yiq(im_orig)
        color_image[:, :, 0] = channel
        color_image = yiq2rgb(color_image)
        return [color_image, image_histogram, hist_eq]


# imdisplay("low_contrast.jpg",GRAYSCALE)

image = read_image("low_contrast.jpg", RGB)

image = histogram_equalize(image)[0]
# plt.plot(np.histogram(image, 256, (0, 1))[0])
# print(image)
# print(image.dtype)
plt.imshow(image, cmap='gray', vmin=0,vmax=1)
plt.show()

