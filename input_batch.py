from __future__ import division

from lunithandler import lflags
import csv, time, random
import tensorflow as tf
import numpy as np
from utils import image_preprocess


def load_data_from_csv(csv_file, shuffle=True):
    """ Load lines in csv file as a list of strings

        Args:
            csv_file: csv file path
            shuffle: boolean, for double shuffling
    """

    with open(csv_file, 'rb') as f:
        lines = [l.strip() for l in f.readlines()]

    if shuffle:
        random.shuffle(lines)

    return lines

def create_data_queue(data_strs, train_mode=True):
    """ Create queue having data as strings

        Args:
            data_strs: list of strings returned from load_data_from_csv
            train_mode: boolean,
                        if True, queue will be shuffled every epoch
    """

    return tf.train.string_input_producer(data_strs,
                    shuffle=train_mode, capacity=len(data_strs))
    

def read_data_test(queue):
    """ For test """

    data_info = queue.dequeue()
    x,y = tf.decode_csv(data_info, [[1],[1]], field_delim=' ')

    return x,y


def read_data(queue, num_im=1, num_label=1, depth=3, img_dtype=tf.uint16):
    """ Dequeue data element and decode as a pair of image and label
    """
    data_info = queue.dequeue()

    imgs = []
    labels = []

    records = tf.decode_csv( data_info, [ [""] for i in xrange( num_im ) ] + [ [1] for i in xrange( num_label ) ], field_delim=' ' ) 
    # list of tensor
    #columns_1 = [[1] for i in xrange(lflags.num_classes[0])] 
    #columns_2 = [[1] for i in xrange(lflags.num_classes[1])] 
    for idx in xrange( num_im ) :
        imgs.append( tf.image.convert_image_dtype( tf.image.decode_jpeg( tf.read_file( records[idx] ), depth ), dtype=tf.float32 ) )        
    for idx in xrange( int( num_label ) ) :
        c_idx = num_im+idx
        labels.append( records[c_idx] ) 
    return imgs, labels

def preprocess_image( imgs, train_mode=True ) :
    """ Preprocess image for random augmentation if train mode,
        or for normalization if test mode.

        Args:
            img: image
            train_mode: boolean
            thread_id: for choosing resize method
    """
    input_size = lflags.input_size 
    crop_size = lflags.crop_size
    processeds = []
    for img in imgs :
        if train_mode :
            processed = image_preprocess.resize_image(img, input_size, preserve_ratio=True)
            processed = processed*255
            if lflags.gamma[0]:
                processed = image_preprocess.random_gamma(processed, max_delta=lflags.gamma[1])
            if lflags.brightness[0]:
                processed = image_preprocess.random_brightness(processed, max_delta=lflags.brightness[1])
            if lflags.contrast[0]:
                lower, upper = lflags.contrast[1:]
                processed = image_preprocess.random_contrast(processed, lower=lower, upper=upper)
            if lflags.blur_or_sharpen[0]:
                min_s, max_s, min_a, max_a = lflags.blur_or_sharpen[1:]
                processed = image_preprocess.random_blur_or_sharpen(processed, (min_s, max_s, min_a, max_a), device_id=thread_id%4)
        else:
            processed = image_preprocess.resize_image(img, input_size, preserve_ratio=True)

        # normalize
        #processed = tf.clip_by_value(processed, 0.0, 1.0) 
        processed = tf.subtract(processed, lflags.channel_mean)

        processed = tf.image.resize_image_with_crop_or_pad(processed, input_size, input_size)
        if train_mode:
            processed = tf.random_crop(processed, [crop_size,crop_size,3])
        else:
            processed = tf.image.resize_image_with_crop_or_pad(processed, crop_size, crop_size)
        processeds.append( processed )
    return processeds

def preprocessed_inputs(fileinfo_queue, batch_size, num_threads=5, train_mode=True):

    with tf.device('/cpu:0'):

        images, labels = read_data( fileinfo_queue, num_im=3, num_label=6 )
        images = preprocess_image( images, train_mode=train_mode )

        if train_mode:
            images, labels = tf.train.shuffle_batch([images,labels], batch_size=batch_size, 
                                                     num_threads=num_threads,  capacity=1000+4*batch_size, 
                                                     min_after_dequeue=1000)
        else:
            images, labels = tf.train.batch([images,labels], batch_size=batch_size, num_threads=num_threads,
                                            capacity=50, allow_smaller_final_batch=True)
        #    images = tf.reshape(images, shape=[batch_size, lflags.crop_size, lflags.crop_size, 1])
        #    labels = tf.reshape(labels, shape=[batch_size, lflags.num_classes])

        #input_list = [batch_inputs(fileinfo_queue, train_mode=train_mode) for _ in range(3)]

        #images, labels = tf.train.shuffle_batch_join(input_list, batch_size=batch_size, 
        #                                         capacity=1000+6*batch_size, 
        #                                         min_after_dequeue=1000)
    return images, labels

if __name__ == '__main__':

    #csv_file = 'inputs/OnlyVS_multiclass_False_idd_True_ratio_0.20_oversample_False_is_anno_True_trn_pilotstudy.csv'
    csv_file = 'inputs/ip_base_trn.csv'
    lflags.DEFINE_integer('input_size', 750, """Target image size before random cropping.""")
    lflags.DEFINE_integer('crop_size', 750, """Image size feeding to network.""")
    lflags.DEFINE_float('channel_mean', 0.0, """Channel mean value.""")
    lflags.DEFINE_bool('reverse_grayscale', False, """Randomly reverse grayscale""")
    lflags.DEFINE_bool('horizontal_flip', False, """Randomly flip image horizontally""")
    lflags.DEFINE_bool('rotation', False, """Randomly rotate image""")
    lflags.DEFINE_list('brightness', [True, 0.1], """Randomly adjust brightness""")
    lflags.DEFINE_list('gamma', [True, 0.25], """Randomly adjust brightness""")
    lflags.DEFINE_list('contrast', [True, 0.7, 1.3], """Randomly adjust contrast""")   
    #lflags.DEFINE_list('blur', [True, 0.5, 1.5], """Randomly adjust contrast""")   
    #lflags.DEFINE_list('sharpen', [True, 0.5, 1.5, 0.5, 1.5], """Randomly adjust contrast""")   
    lflags.DEFINE_list('blur_or_sharpen', [True, 0.3, 0.6, 1.0, 2.0], """Randomly adjust contrast""")   
 
    lflags.DEFINE_integer('batch_size', 16, 
                                """Number of images to process in a batch.""")
   
    with tf.device('/cpu:0'): 
        data_strs = load_data_from_csv(csv_file, shuffle=True)
        #imgs, labels = preprocessed_inputs(data_strs,mode='train')
        imgs, labels = preprocessed_inputs(data_strs, train_mode=True)

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    aa,bb = sess.run([imgs,labels])

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    for i in xrange(10):
        plt.subplot(1,10,i)
        plt.imshow(aa[i,:,:,0],cm.Greys_r,vmin=0.0,vmax=1.0)
    plt.show()

    import ipdb
    ipdb.set_trace()
