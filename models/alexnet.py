import tensorflow as tf
import numpy as np
import sys
from network import *

TOWER_NAME = 'alexnet'

def alexnet( _X, _dropout ):
    # TODO weight decay loss tern
    # Layer 1 (conv-relu-pool-lrn)
    conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    # Layer 2 (conv-relu-pool-lrn)
    conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
    conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
    norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    # Layer 3 (conv-relu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
    # Layer 4 (conv-relu)
    conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
    # Layer 5 (conv-relu-pool)
    conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
    # Layer 6 (fc-relu-drop)
    fc6 = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
    fc6 = dropout(fc6, _dropout)
    # Layer 7 (fc-relu-drop)
    fc7 = fc(fc6, 4096, 4096, name='fc7')
    fc7 = dropout(fc7, _dropout)
    # Layer 8 (fc-prob)
    fc8_1 = fc(fc7, 4096, 17, relu=False, name='fc8_1')
    fc8_2 = fc(fc7, 4096, 20, relu=False, name='fc8_2' )
    return fc6, fc7, fc8_1, fc8_2

def loss( fc6, fc7, fc8_1, fc8_2, labels_1, labels_2 ) :
    fc6 = tf.split( fc6, 3 )
    fc7 = tf.split( fc7, 3 )
    
    with tf.op_scope( [ fc7 ], 'loss_dist', 'L2LOSS' ) :
        similar_image_l2        = tf.sqrt( tf.reduce_sum( tf.pow( tf.subtract( fc7[0], fc7[1] ), 2 ), 1 ) )
        unsimilar_image_l2      = tf.sqrt( tf.reduce_sum( tf.pow( tf.subtract( fc7[0], fc7[2] ), 2 ), 1 ) )
        loss_dist = tf.reduce_mean( tf.maximum( tf.add( tf.subtract( similar_image_l2, unsimilar_image_l2 ), 0.3 ), 0 ), name='value' )
        tf.add_to_collection( 'losses', loss_dist ) 
    
    tf.contrib.losses.softmax_cross_entropy( fc8_1, labels_1, scope='loss_cls_1')
    tf.contrib.losses.softmax_cross_entropy( fc8_2, labels_2, scope='loss_cls_2')
