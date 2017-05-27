from __future__ import absolute_import
from __future__ import division
import sys
sys.path.append('./utils')

import logging

from multigpu import average_gradients

from lunithandler import lflags
#import gflags
import cPickle
import tensorflow as tf
import numpy as np
import time
import os
import re
import csv
import importlib
import input_batch
#import input_batch_old as input_batch

from tensorflow.python.client import timeline
#FLAGS = gflags.FLAGS

model = importlib.import_module(lflags.model)

#import ipdb
#ipdb.set_trace()
LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(filename='%s/%s.log'%(lflags.train_dir,lflags.trial+'_trn'),filemode='a',level=logging.INFO,format=LOG_FORMAT)

def train_optimizer(learning_rate):

    if lflags.optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, lflags.decay, 
                momentum=lflags.momentum, epsilon=lflags.epsilon)
    elif lflags.optimizer == 'sgd_with_momentum':
        return tf.train.MomentumOptimizer(learning_rate, lflags.momentum, use_nesterov=lflags.use_nesterov)

def tower_loss(fileinfo_queue_list, n_batches_per_tower, alpha, scope):
    """
        Args:
            alpha: weight for losses, (1-alpha)*loss_cls + alpha*loss_loc
    """
    images_list = []
    labels_list = []
    for idx, fileinfo_queue in enumerate(fileinfo_queue_list):
        images, labels = input_batch.preprocessed_inputs(fileinfo_queue, n_batches_per_tower[idx], num_threads=5, with_mask=lflags.with_mask[idx], train_mode=True)
        if not lflags.with_mask[idx] and any(lflags.with_mask): 
            images = tf.pad(images, [[0,0],[0,0],[0,0],[0,lflags.num_classes]])
        images_list.append(images)
        labels_list.append(labels)
    images = tf.concat(images_list,0)
    labels = tf.concat(labels_list,0)
    # Build graph for single tower, which processes batches individually
    logits_cls, pos_map, eff_size_list = model.inference(images[:,:,:,:1],num_classes=lflags.num_classes,for_training=True,scope=scope)
    
    model.loss(logits_cls, labels, pos_map, images, eff_size_list)


    losses_cls = tf.get_collection(tf.GraphKeys.LOSSES, scope+'loss_cls')
    losses_loc = tf.get_collection(tf.GraphKeys.LOSSES, scope+'loss_loc')
   
    losses_cls = tf.add_n(losses_cls, name='loss_cls')
    losses_loc = tf.add_n(losses_loc, name='loss_loc')
    
    losses = alpha*losses_cls + (1.0-alpha)*losses_loc

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
   
    total_loss = tf.add_n([losses]+regularization_losses, name='total_loss')
    
    #for l in [losses] + [total_loss]:
    #    loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    #    tf.scalar_summary(loss_name, l)
    
    #with tf.control_dependencies([total_loss]):
    #    total_loss = tf.identity(total_loss)

    return total_loss, losses, losses_cls, losses_loc


def train():
    
    graph = tf.Graph()
    
    with graph.as_default(), tf.device('/cpu:0'):
        
        global_step = tf.get_variable('global_step', [], dtype=tf.int64, 
                initializer=tf.constant_initializer(0), trainable=False)

        data_strs_list = [input_batch.load_data_from_csv(csv_name, shuffle=True) for csv_name in lflags.train_csv]
        fileinfo_queue_list = [input_batch.create_data_queue(data_strs, train_mode=True) for data_strs in data_strs_list]
        #images, labels = input_batch.preprocessed_inputs(data_strs, train_mode=True)
        
        # Epochs are calculated using the first csv file length & batch size
        num_examples = len(data_strs_list[0])
        num_batches_for_epoch = int(np.ceil(num_examples / lflags.batch_size_list[0]))

        n_batch_per_tower = [int(temp_batch_size/len(lflags.gpu_ids)) for temp_batch_size in lflags.batch_size_list]

        # placeholder for learning rate
        lr = tf.placeholder(tf.float32, shape=[])
        optimizer = train_optimizer(lr)

        # placeholder for weight alpha
        alpha = tf.placeholder(tf.float32, shape=[])

        #images_splits = tf.split(images, len(lflags.gpu_ids), 0)
        #labels_splits = tf.split(labels, len(lflags.gpu_ids), 0)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for idx, i in enumerate(lflags.gpu_ids):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:

                        #cost, loss, loss_cls, loss_loc  = tower_loss(images_splits[idx], labels_splits[idx], alpha, scope)
                        cost, loss, loss_cls, loss_loc = tower_loss(fileinfo_queue_list, n_batch_per_tower, alpha, scope)
  

                        #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        #batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,scope)
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope)
                        grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)
                        
                        tf.get_variable_scope().reuse_variables()

        grads = average_gradients(tower_grads)

        #for grad, var in grads:
        #    if grad is not None:
        #        summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        #for var in tf.trainable_variables():
        #    summaries.append(tf.histogram_summary(var.op.name, var))

        # NOTE: diable moving average op on trainable variables
        #variable_averages = tf.train.ExponentialMovingAverage(0.0, global_step)
        #variables_averages_op = variable_averages.apply(tf.trainable_variables()+tf.moving_average_variables())
        #variables_averages_op = variable_averages.apply(tf.trainable_variables())

        batchnorm_updates_op = tf.group(*batchnorm_updates)
        #train_op = tf.group(apply_gradient_op,variables_averages_op,batchnorm_updates_op)
        train_op = tf.group(apply_gradient_op,batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

        #summary_op = tf.merge_summary(summaries)

        # Initializing the variables
        #init = tf.initialize_all_variables()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=tf.GPUOptions(allow_growth=True)))
        #init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init)

        #coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)

        graph_def = sess.graph.as_graph_def(add_shapes=True)
        #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def = graph_def)
        #summary_writer = tf.train.SummaryWriter(lflags.train_dir, graph = graph_def)

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
            _, cost_value, loss_value, loss_cls_value, loss_loc_value = sess.run([train_op,cost,loss,loss_cls,loss_loc],
                                                                # options=run_options, run_metadata=run_metadata,
                                                                feed_dict={lr: current_lr, alpha: current_alpha})
#             tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('timeline.json', 'w') as f:
                # f.write(ctf)
            duration = time.time() - start_time
            print step, current_lr, current_alpha, loss_value, loss_cls_value, loss_loc_value, duration
            logging.info('[iter] %s, [lr] %s, [alpha] %s, [loss_total] %.4f, [loss_cls] %.4f, [loss_loc] %.4f, [time] %.4f'% \
                         (step, current_lr, current_alpha, loss_value, loss_cls_value, loss_loc_value, duration))
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
