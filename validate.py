from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import ipdb
import os, glob
import importlib
import logging

from sklearn import metrics
import numpy as np
import tensorflow as tf

import input_batch

from lunithandler import lflags


LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(filename='%s/%s.log'%(lflags.train_dir,lflags.trial+'_val'),filemode='a',level=logging.INFO,format=LOG_FORMAT)

model = importlib.import_module(lflags.model)

def eval_once(trained, saver, res_ops, num_iter):
    """ Run evaluation for a trained model

        Args:
            trained: trained model
            saver: Saver
            res_ops: ops for evaluation
            num_iter: number of iterations
    """
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        saver.restore(sess, trained)
        global_step = trained.split('/')[-1].split('-')[-1]

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

            total_sample_count = num_iter * lflags.batch_size
            step = 0
            total_probs = []
            total_labels = []
            while step < num_iter and not coord.should_stop():
                batch_labels, batch_probs = sess.run(res_ops)
                
                total_probs.append(batch_probs)
                total_labels.append(batch_labels)

                step += 1
                if step % 100 == 0:
                    print(step, num_iter)

            total_probs = np.vstack(total_probs)
            total_labels = np.vstack(total_labels)
    
            if lflags.num_classes == 1:
                pred_labels = total_probs > 0.5
                pred_pos_probs = total_probs
            else:
                pred_labels = np.argmax(total_probs,1) 
                pred_pos_probs = total_probs[:,1]

            top_1_accu = metrics.accuracy_score(total_labels[:,1], pred_labels)
            fpr, tpr, thresholds = metrics.roc_curve(total_labels[:,1], pred_pos_probs, pos_label=1)
            auc_value = metrics.auc(fpr,tpr)

            # calculate sensitivity and specificity at threshold 0.5
            confusion_mat = metrics.confusion_matrix(total_labels[:,1], pred_pos_probs>=0.5)
            sensitivity = confusion_mat[1,1] / (confusion_mat[1,1]+confusion_mat[1,0])
            specificity = confusion_mat[0,0] / (confusion_mat[0,0]+confusion_mat[0,1])

            print(trained)
            print('Accuracy: %.4f, AUC: %.4f' % (top_1_accu, auc_value))
            logging.info('[iter] %s, [accu] %.4f, [AUC] %.4f, [Sens.@0.5] %.4f, [Spec.@0.5] %.4f' % \
                         (global_step, top_1_accu, auc_value, sensitivity, specificity))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    return total_labels, total_probs, (top_1_accu,auc_value)

def evaluate(trained):
    """  
    """
    graph = tf.Graph()

    with graph.as_default():
        
        data_strs = input_batch.load_data_from_csv(lflags.val_csv,shuffle=False)
        num_examples = len(data_strs)

        data_paths = [i.split(',')[0] for i in data_strs]
        num_batches_for_epoch = int(np.ceil(num_examples / lflags.batch_size))

        fileinfo_queue = input_batch.create_data_queue(data_strs, train_mode=False)
        images, labels = input_batch.preprocessed_inputs(fileinfo_queue,1,train_mode=False)

        with tf.device('/gpu:{}'.format(lflags.gpu_id)):
            _, logits, _ = model.inference(images,lflags.num_classes,for_training=False)

        if lflags.num_classes == 1:
            probs = tf.sigmoid(logits)
        else:
            probs = tf.nn.softmax(logits)

        saver = tf.train.Saver(tf.all_variables())

        true_labels, probs, res = eval_once(trained, saver, [labels,probs], num_batches_for_epoch)
        return data_paths, true_labels, probs, res

def validate():
    """ main for validation
    """

    trained_models = []
    for trained in glob.glob(os.path.join(lflags.train_dir, '*ckpt*')):
        if 'index' in trained:
            trained = os.path.splitext(trained)[0]

            n_iter = int(trained.split('/')[-1].split('-')[-1])
            trained_models.append((n_iter, trained))

    trained_models.sort(key=lambda x:x[0])
    trained_models = trained_models[-5:]

    for model in trained_models:
        data_paths, true_labels, probs, res = evaluate(model[1])

