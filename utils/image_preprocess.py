import tensorflow as tf
import numpy as np

distance_matrix = None

def get_dist_matrix():

    global distance_matrix

    # distance matrix for blur process
    if type(distance_matrix)!=np.ndarray:
        kernel_radius = 3
        kernel_size = int(np.ceil(6*kernel_radius))
        kernel_size = kernel_size + 1 if kernel_size%2==0 else kernel_size
        distance_matrix = np.zeros([kernel_size, kernel_size])
        center = kernel_size/2
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance_matrix[i,j]= ((center-i)**2 + (center-j)**2)**.5
        distance_matrix = np.expand_dims(distance_matrix,2)
        distance_matrix = np.expand_dims(distance_matrix,3)

    return distance_matrix

def subtracttract_mean(image, mean_value):

    return image - mean_value

def reverse_grayscale(image):
    """ Reverse pixel values
    """

    return tf.reduce_max(image)-(image-tf.reduce_min(image))

def random_reverse_grayscale(image):

    rand_value = tf.random_uniform([])
    rand_cond = tf.greater_equal(rand_value,0.5)

    return tf.cond(rand_cond, lambda: reverse_grayscale(image),
                              lambda: image)

def random_brightness(image, max_delta, always=False):

    if always:
        return tf.image.random_brightness(image, max_delta=max_delta)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)

        return tf.cond(rand_cond, lambda: tf.image.random_brightness(image, max_delta=max_delta),
                                  lambda: image)

def random_contrast(image, lower, upper, always=False):

    if always:
        return tf.image.random_contrast(image, lower=lower, upper=upper)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)

        return tf.cond(rand_cond, lambda: tf.image.random_contrast(image, lower=lower, upper=upper),
                                  lambda: image)

def adjust_gamma(image, gamma=1.0, gain=1.0):

    #NOTE: pixel values should lie within [0,1]
    return image ** gamma * gain

def random_gamma(image, max_delta, gain=1, always=False):

    def _random_adjust(image, max_delta, gain=1):
        rand_gamma = tf.random_uniform([], minval=1.0-max_delta, maxval=1.0+max_delta)
        return adjust_gamma(image, gamma=rand_gamma, gain=gain)

    if always:
        return _random_adjust(image, max_delta=max_delta, gain=gain)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image, max_delta=max_delta, gain=gain),
                                  lambda: image)

def blur(image, sigma, device_id):

    dist = get_dist_matrix()
    kernel = tf.exp(-dist/(2*sigma**2))
    kernel = kernel/tf.reduce_sum(kernel)
    image = tf.expand_dims(image,0)
    with tf.device('/gpu:%d'%device_id):
        processed = tf.nn.conv2d(image, kernel, [1,1,1,1], 'SAME')
    processed = tf.squeeze(processed, axis=0)

    return processed

def random_blur(image, min_sigma, max_sigma, device_id, always=True):
    
    def _random_adjust(image):
        sigma = tf.random_uniform([], minval=min_sigma, maxval=max_sigma)
        return blur(image, sigma, device_id)

    if always:
        return _random_adjust(image)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image),
                                  lambda: image)
        
def sharpen(image, sigma, amount, device_id):
    blurred = blur(image, sigma, device_id)
    processed = image + (image - blurred) * amount

    return processed

def random_sharpen(image, min_sigma, max_sigma, min_amount, max_amount, device_id, always=True):
    
    def _random_adjust(image):
        sigma = tf.random_uniform([], minval=min_sigma, maxval=max_sigma)
        amount = tf.random_uniform([], minval=min_amount, maxval=max_amount)
        return sharpen(image, sigma, amount, device_id)

    if always:
        return _random_adjust(image)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image),
                                  lambda: image)

def random_blur_or_sharpen(image, params, device_id):
    sigma = tf.random_uniform([], minval=params[0], maxval=params[1])
    amount = tf.random_uniform([], minval=params[2], maxval=params[3])

    rand_value = tf.random_uniform([])
    rand_cond = tf.greater_equal(rand_value, 0.5)

    return tf.cond(rand_cond, lambda: blur(image, sigma, device_id), lambda: sharpen(image, sigma, amount, device_id))

def augment_rotation(image, degree=0):
    """ Rotate image

        Args:
            degree: 0(original), 1(90), 2(180), 3(270)
    """

    if degree == 0:
        return image
    elif degree == 1:
        rotated = tf.image.flip_up_down(image)
        rotated = tf.transpose(rotated, [1,0,2])
        return rotated
    elif degree == 2:
        rotated = tf.image.flip_left_right(image)
        rotated = tf.image.flip_up_down(rotated)
        return rotated
    else:
        rotated = tf.transpose(image, [1,0,2])
        rotated = tf.image.flip_up_down(rotated)
        return rotated


def resize_image(image, target_size, method=0, preserve_ratio=False):
    """ Resize image with original aspect ratio

        Args:
            target_size: target size of longer axis
            method: bilinear(0), nearest_neighbor(1), bicubic(2), area(3)
            preserve_ratio: boolean
    """
    image_shape = tf.shape(image)
    original_height = tf.cast(image_shape[0],dtype=tf.float32)
    original_width = tf.cast(image_shape[1],dtype=tf.float32)

    target_size = tf.cast(target_size,dtype=tf.float32)

    if preserve_ratio:
        target_h_and_w = tf.cond(
            tf.greater_equal(original_height, original_width),
            lambda: (target_size, tf.div(original_width,original_height)*target_size),
            lambda: (tf.div(original_height,original_width)*target_size, target_size))
    else:
        target_h_and_w = [target_size,target_size]

    target_h_and_w[0] = tf.cast(target_h_and_w[0],dtype=tf.int32)
    target_h_and_w[1] = tf.cast(target_h_and_w[1],dtype=tf.int32)

    resized = tf.expand_dims(image,0)

    try:
        resized = tf.image.resize_images(resized,target_h_and_w[0],target_h_and_w[1],method=method)
    except:
        resized = tf.image.resize_bilinear(resized,tf.stack(target_h_and_w))
    resized = tf.squeeze(resized,[0])

    # NOTE: tf.image.resize_images does not work!! --> fixed?
    #resized = tf.image.resize_images(image,target_h_and_w[0],target_h_and_w[1])

    return resized

def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
    cropped = tf.slice(
        image,
        tf.stack([offset_height, offset_width, 0]),
        tf.stack([target_height, target_width, -1]))

    return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    after_padding_width = tf.subtract(
        tf.subtract(target_width, offset_width),  original_width)
    after_padding_height = tf.subtract(
        tf.subtract(target_height, offset_height), original_height)

    paddings = tf.reshape(
        tf.stack(
            [offset_height, after_padding_height,
             offset_width, after_padding_width,
             0, 0]), [3, 2])

    padded = tf.pad(image, paddings)

    return padded


def resize_image_with_crop_or_pad(image, target_height, target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    target_height = tf.cast(target_height,dtype=tf.int32)
    target_width = tf.cast(target_width,dtype=tf.int32)

    zero = tf.constant(0)
    half = tf.constant(2)

    offset_crop_width = tf.cond(
        tf.less(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.subtract(original_width, target_width), half),
        lambda: zero)

    offset_pad_width = tf.cond(
        tf.greater(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.subtract(target_width, original_width), half),
        lambda: zero)

    offset_crop_height = tf.cond(
        tf.less(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.subtract(original_height, target_height), half),
        lambda: zero)

    offset_pad_height = tf.cond(
        tf.greater(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.subtract(target_height, original_height), half),
        lambda: zero)

    cropped = crop_to_bounding_box(
        image, offset_crop_height, offset_crop_width,
        tf.minimum(target_height, original_height),
        tf.minimum(target_width, original_width))

    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)

    return resized

def pad_and_crop_image_dimensions(target_height, target_width, image_dir):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(
            "{image_dir}/*.jpg".format(image_dir=image_dir)))

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)

    resized_padding = resize_image_with_crop_or_pad(image, target_height + 1,
                                                    target_width + 1)
    resized_cropping = resize_image_with_crop_or_pad(image, target_height - 1,
                                                     target_width - 1)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        result = sess.run([image, resized_padding, resized_cropping])
        yield(result[0], result[1], result[2])

        coord.request_stop()
        coord.join(threads)
