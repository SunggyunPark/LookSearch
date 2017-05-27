from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os, glob
sys.path.append('./utils')
sys.path.append('/lunit/home/sgpark/projects/libs/lunit-handler')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import visualize
import calculate_metrics
from lunithandler import lflags
import input_batch
import logging

from datetime import datetime
import math
import time
import ipdb
import importlib
import cPickle
import pandas as pd

from sklearn import metrics
import numpy as np

import skimage
import skimage.io
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
plt.switch_backend('agg')


GPU_ID = 0

lflags.DEFINE_string( 'model', 'models.resnet_ss', """ """ )
lflags.DEFINE_string( 'trial',  'test_ss',""" """ )
lflags.DEFINE_integer( 'num_classes', 2,""" """)
lflags.DEFINE_list( 'models', [ 'test_ss/model.ckpt-100100' ], """List of models""" )
lflags.DEFINE_list( 'archs', [ 'resnet_ss' ], """List of archs""" )
lflags.DEFINE_integer( 'input_size', 1024, """ """ )
lflags.DEFINE_integer( 'crop_size', 1024, """ """ )
lflags.DEFINE_list( 'input_sizes', [1000, 950, 900, 850, 800, 750, 700, 650, 600], """List of input size""")
lflags.DEFINE_string( 'test_csv', 'inputs/lung_cancer_val.csv', """Either 'test' or 'train_eval'.""" )
lflags.DEFINE_integer( 'batch_size', 1, """ """ )
lflags.DEFINE_float('channel_mean', 0.6, """ """)
lflags.DEFINE_float('weight_decay', 0.0001, """ """)
lflags.DEFINE_string( 'archive',  '/lunit/archive/sgpark',""" """ )
LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig( filename='%s/%s/%s.log' %( lflags.archive, lflags.trial, lflags.trial+'_tst' ), filemode='a', level=logging.INFO, format=LOG_FORMAT )

def preprocess_img( im_path, im_mask=False ) :
    ims = []
    im = skimage.img_as_float( skimage.io.imread( im_path ) ).astype( np.float32 ) 
    h,w = im.shape
    temp_im = np.zeros( ( lflags.input_size, lflags.input_size ) )
    if max( h, w ) < lflags.input_size :
        temp_im[ int( (lflags.input_size-h)/2 ):int( (lflags.input_size-h)/2 )+h,
                    int( (lflags.input_size-w)/2 ):int( (lflags.input_size-w)/2 )+w] = im
    else :
        if h>w :
            if w > lflags.input_size :
                temp_im = im[ int( (h-lflags.input_size)/2 ):int( (h-lflags.input_size)/2 )+lflags.input_size,
                    int( (w-lflags.input_size)/2 ):int( (w-lflags.input_size)/2 )+lflags.input_size ]
            else :
                temp_im[ :,  int( (lflags.input_size-w)/2 ):int( (lflags.input_size-w)/2 )+w ]=im[ int( (h-lflags.input_size)/2 ):int( (h-lflags.input_size)/2 )+lflags.input_size, : ]
        else :
            if h > lflags.input_size :
                temp_im = im[ int( (h-lflags.input_size)/2 ):int( (h-lflags.input_size)/2 )+lflags.input_size,
                    int( (w-lflags.input_size)/2 ):int( (w-lflags.input_size)/2 )+lflags.input_size ]
            else :
                temp_im[ int( (lflags.input_size-h)/2 ):int( (lflags.input_size-h)/2 )+h, : ]=im[ :, int( (w-lflags.input_size)/2 ):int( (w-lflags.input_size)/2 )+lflags.input_size, : ]
    temp_im = np.copy( temp_im )                   
    if not im_mask :
        ims.append( temp_im )
        for input_size in lflags.input_sizes :
            resize_im = skimage.transform.resize( temp_im, ( input_size, input_size ), mode='constant', preserve_range=True )
            ims.append( resize_im )
        return ims
    else :  
        return temp_im
    return ims

def build_graph( architecture, trained_model ) :
    import importlib
    model = importlib.import_module( 'models.'+architecture )
    trained_model = os.path.join( lflags.archive, trained_model )
    
    with tf.Graph().as_default() :
        with tf.device( '/gpu:%d'%GPU_ID ) :
            test_img = tf.placeholder( tf.float32 )
            #test_img.set_shape( [ None, input_size, input_size, 1 ] )
            test_img.set_shape( [ None, None, None, 1 ] )
            logit, pos_map, _ = model.inference( test_img, lflags.num_classes, for_training=False, visualize=True )
            prob = tf.nn.softmax( logit )
            pos_map = tf.transpose( pos_map,[0,2,3,1] )
        sess = tf.Session( config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth=True)))
        saver = tf.train.Saver( tf.all_variables() )
        saver.restore( sess, trained_model )
    return sess, ( test_img, prob, pos_map )

def inference( im, in_out, sess ) :
    test_img, prob, pos_map = in_out 
    prob, pos_map = sess.run( [prob, pos_map], feed_dict={ test_img:im } )
    try :
        pos_prob = prob[0][1]
    except : 
        pos_prob = prob[1]
    pos_map  = pos_map[0, :, :, 1]
    return pos_prob, pos_map 


def evaluate() :
    import ipdb
    ipdb.set_trace()
    graphs = []
    in_outs = []
    models_for_ensemble = zip( lflags.archs, lflags.models )
    for idx, model in enumerate( models_for_ensemble ) :
        tmp_arch, tmp_model = model
        tmp_graph, in_out = build_graph( tmp_arch, tmp_model )
        graphs.append( tmp_graph )
        in_outs.append( in_out )

    data_strs = input_batch.load_data_from_csv( lflags.test_csv, shuffle=False )
    data_paths = [ i.split()[0] for i in data_strs ]
    labels = [ i.split()[1] for i in data_strs ] 
 
    data_paths = data_paths[0:3]
    labels = labels[0:3]
 
    batch_pos_probs = []
    batch_pos_maps  = []
    batch_labels    = []
    for idx, info in enumerate( zip( labels, data_paths ) ):
        if idx%10 == 0 :
            print('[%s] [Prediction] %d of %d done...' %( lflags.trial, idx, len( labels ) ) )
        label, im_path = info
        if label == '0,1' :
            batch_labels.append( 1 )
        else :
            batch_labels.append( 0 )
        ims = preprocess_img( im_path )
        final_pos_map = []
        final_pos_prob = []
        for gidx, session in enumerate( graphs ) :
            input_pos_map = []
            input_pos_prob = []
            for sidx, temp_im in enumerate( ims ) :
                temp_im = temp_im-0.6
                temp_im = np.reshape( temp_im, ( 1, temp_im.shape[0], temp_im.shape[1], 1 ) )          
                pos_prob, pos_map = inference( temp_im, in_outs[gidx], session ) 
                pos_map = skimage.transform.resize( pos_map, ( 100, 100 ), preserve_range=True, mode='constant' )
                centered_pos_map = pos_map - np.min( pos_map )
                norm_activations = centered_pos_map / ( np.max( pos_map )-np.min( pos_map ) ) 
                input_pos_map.append( norm_activations*pos_prob )
                input_pos_prob.append( pos_prob )
            final_pos_map.append( np.mean( input_pos_map, axis=0 ) )
            final_pos_prob.append( np.mean( input_pos_prob ) )
        batch_pos_maps.append( np.mean( final_pos_map, axis=0 ) )      
        batch_pos_probs.append( np.mean( final_pos_prob ) )
    map_path = '/lunit/data/cxr/validation/val_lung_cancer_lesion/'
    overlap = 1
    im_masks = []
    for im_path in data_paths :
        im_mask_path = os.path.join( map_path, im_path.split('/')[-1].replace('.png',''), '%02d.png' %overlap )
        im_mask = preprocess_img( im_mask_path, im_mask=True )         
        im_mask = skimage.transform.resize( im_mask, ( 100, 100 ), preserve_range=True, mode='constant', order=0 )
        im_masks.append( im_mask )
    auc, acc, jafroc, sensitivity, specificity = calculate_metrics.calculate_metrics( batch_pos_maps, batch_pos_probs, im_masks )
    #dst_path = os.path.join( lflags.archive, lflags.trial )
    #top1_acc, auc, loc_ap, sensitivity, specificity = calculate_metrics.calcuate_metrics( data_paths, batch_pos_maps, batch_pos_probs, batch_labels, map_path, dst_path, overlap=1 ) 
    logging.info('[exp] %s, [models] %s [archs] %s, [input_sizes] %s\n                      [accu] %.4f, [AUC] %.4f, [LOC AP] %.4f, [Sens.@0.5] %.4f, [Spec.@0.5] %.4f'
                     %( lflags.trial, lflags.models, lflags.archs, lflags.input_sizes, top1_acc, auc, loc_ap, sensitivity, specificity ) ) 


#def main( ) :
#    
##    with open( '/data3/cxr/db/lung_cancer_db.pkl', 'r' ) as f :
##        lc = cPickle.load( f )
##    with open( '/data3/cxr/db/metastasis_db.pkl', 'r' ) as f :
##        mt = cPickle.load( f )
##    with open( '/data3/cxr/db/normal_db.pkl', 'r' ) as f :
##        no = cPickle.load( f )
##    df = pd.concat( [ lc, mt, no ], ignore_index=True )
#    evaluate()

if __name__ == '__main__':
    evaluate()
