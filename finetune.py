from __future__ import absolute_import
from __future__ import division
import sys
sys.path.append('./utils')

import logging

from multigpu import average_gradients

from lunithandler import lflags
import cPickle
import tensorflow as tf
import numpy as np
import time
import os
import re
import csv
import importlib
import input_batch
from network import *

from tensorflow.python.client import timeline

model = importlib.import_module(lflags.model)

LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(filename='%s/%s.log'%(lflags.train_dir,lflags.trial+'_trn'),filemode='a',level=logging.INFO,format=LOG_FORMAT)
preweights = '/lunit/home/sgpark/data/pretrained_net/pretrained_alexnet.npy'

def train_optimizer(learning_rate):

    if lflags.optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, lflags.decay, 
                momentum=lflags.momentum, epsilon=lflags.epsilon)
    elif lflags.optimizer == 'sgd_with_momentum':
        return tf.train.MomentumOptimizer(learning_rate, lflags.momentum, use_nesterov=lflags.use_nesterov)

def tower_loss(fileinfo_queue, n_batches_per_tower, alpha, dropout, scope):
    """
        Args:
            alpha: weight for losses, (1-alpha)*loss_cls + alpha*loss_loc
    """
    images, labels = input_batch.preprocessed_inputs(fileinfo_queue, n_batches_per_tower, num_threads=5, train_mode=True)

    # Build graph for single tower, which processes batches individually
    images = tf.concat( [ images[:,0,:,:], images[:,1,:,:], images[:,2,:,:] ], 0  )
    
    # Load Caffe net --> RGB TO BGR
    channels = tf.unstack (images, axis=-1)
    images    = tf.stack   ([channels[2], channels[1], channels[0]], axis=-1)
    
    labels = tf.concat( [ labels[:,0:2], labels[:,2:4], labels[:,4:6] ], 0 )
    labels_1, labels_2 = tf.split( labels, 2, 1 )  
    labels_1 = tf.one_hot( labels_1, lflags.num_classes[0] )[:, 0, :] 
    labels_2 = tf.one_hot( labels_2, lflags.num_classes[1] )[:, 0, :]

    fc6, fc7, fc8_1, fc8_2 = model.alexnet( images, dropout )

    model.loss( fc6, fc7, fc8_1, fc8_2, labels_1, labels_2 )
    
    # Dist means vector distance
    loss_dist = tf.get_collection( tf.GraphKeys.LOSSES, scope+'loss_dist' )
    loss_cls_1 = tf.get_collection( tf.GraphKeys.LOSSES, scope+'loss_cls_1' )
    loss_cls_2 = tf.get_collection( tf.GraphKeys.LOSSES, scope+'loss_cls_2' )
   
    loss_dist = tf.add_n( loss_dist, name='loss_dist')
    loss_cls_1 = tf.add_n( loss_cls_1, name='loss_cls_1')
    loss_cls_2 = tf.add_n( loss_cls_2, name='loss_cls_2')
    
    losses = alpha*loss_dist + (1.0-alpha)*( loss_cls_1+loss_cls_2 )

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
   
    total_loss = tf.add_n([losses]+regularization_losses, name='total_loss')
    
    return total_loss, losses, loss_dist, loss_cls_1, loss_cls_2


def train():
    
    graph = tf.Graph()
    
    with graph.as_default(), tf.device('/cpu:0'):
        
        global_step = tf.get_variable('global_step', [], dtype=tf.int64, 
                initializer=tf.constant_initializer(0), trainable=False)

        data_strs = input_batch.load_data_from_csv( lflags.train_csv, shuffle=True )
        fileinfo_queue = input_batch.create_data_queue(data_strs, train_mode=True)
         
        # Epochs are calculated using the first csv file length & batch size
        num_examples = len(data_strs)
        num_batches_for_epoch = int(np.ceil(num_examples/lflags.batch_size))

        n_batch_per_tower = int(lflags.batch_size/len(lflags.gpu_ids))

        # placeholder for learning rate
        lr = tf.placeholder(tf.float32, shape=[])
        optimizer = train_optimizer(lr)

        # placeholder for weight alpha
        alpha = tf.placeholder(tf.float32, shape=[])
        
        # placeholder for dropout
        dropout = tf.placeholder( tf.float32 )
        
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope() ) as scope:
            for idx, i in enumerate(lflags.gpu_ids):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        
                        cost, loss, loss_dist, loss_cls_1, loss_cls_2 = tower_loss(fileinfo_queue, n_batch_per_tower, alpha, dropout, scope)
                        grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)
                        
                        tf.get_variable_scope().reuse_variables()
        grads = average_gradients(tower_grads)

        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

        # Initializing the variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(init)

        load_with_skip_siamense( preweights, sess, ['fc8'] )
        print "***"
        print( "    Model restored: {}".format(preweights))
        print "***"
        
        threads = tf.train.start_queue_runners(sess=sess)
        
        graph_def = sess.graph.as_graph_def(add_shapes=True)

        epoch_schedule = lflags.epoch_schedule
        max_epochs = epoch_schedule[-1]
        iter_schedule = np.array(epoch_schedule) * num_batches_for_epoch
        lr_schedule = lflags.lr_schedule
       
        alpha_schedule = lflags.alpha_schedule
        
        for step in xrange(max_epochs*num_batches_for_epoch+1):
            current_idx = np.argmax(step<=iter_schedule)
            current_lr = lr_schedule[current_idx]
            current_alpha = alpha_schedule[current_idx]
            
            start_time = time.time()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, cost_value, loss_value, loss_dist_value, loss_cls_1_value, loss_cls_2_value = sess.run( [ train_op, cost, loss, loss_dist, loss_cls_1, loss_cls_2 ],
                                                                feed_dict={ lr: current_lr, alpha: current_alpha, dropout:1.0 } )
            duration = time.time() - start_time
            print '[%05d/%05d]( lr:%.3f, alpha:%.3f ) total loss : %.5f, dist loss : %.5f, cls_1 loss : %.5f, cls_2 loss : %.5f, duration : %.3fs/im' %( step, max_epochs*num_batches_for_epoch, current_lr, current_alpha, loss_value, loss_dist_value, loss_cls_1_value, loss_cls_2_value, duration )
            logging.info('[iter] %s, [lr] %s, [alpha] %s, [loss_total] %.4f, [loss_dist] %.4f, [loss_cls_1] %.4d, [loss_cls_2] %.4f, [time] %.4f'% \
                         (step, current_lr, current_alpha, loss_value, loss_dist_value, loss_cls_1_value, loss_cls_2_value, duration))
            #if step % 300 == 0:
                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str,step)

            if step % (num_batches_for_epoch*lflags.save_step) == 0 and step !=0:
                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(lflags.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
