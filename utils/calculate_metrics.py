from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import os

import skimage
import skimage.io
import skimage.transform

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
plt.switch_backend('agg')

def preprocess_img( im_path, input_size, input_resize ) :
    ims = []
    im = skimage.img_as_float( skimage.io.imread( im_path ) ).astype( np.float32 ) 
    h,w = im.shape
    temp_im = np.zeros( ( input_size, input_size ) )
    if max( h, w ) < input_size :
        temp_im[ int( (input_size-h)/2 ):int( (input_size-h)/2 )+h,
                    int( (input_size-w)/2 ):int( (input_size-w)/2 )+w] = im
    else :
        if h>w :
            if w > input_size :
                temp_im = im[ int( (h-input_size)/2 ):int( (h-input_size)/2 )+input_size,
                    int( (w-input_size)/2 ):int( (w-input_size)/2 )+input_size ]
            else :
                temp_im[ :,  int( (input_size-w)/2 ):int( (input_size-w)/2 )+w ]=im[ int( (h-input_size)/2 ):int( (h-input_size)/2 )+input_size, : ]
        else :
            if h > input_size :
                temp_im = im[ int( (h-input_size)/2 ):int( (h-input_size)/2 )+input_size,
                    int( (w-input_size)/2 ):int( (w-input_size)/2 )+input_size ]
            else :
                temp_im[ int( (input_size-h)/2 ):int( (input_size-h)/2 )+h, : ]=im[ :, int( (w-input_size)/2 ):int( (w-input_size)/2 )+input_size, : ]
    temp_im = np.copy( temp_im )                   
    im = skimage.transform.resize( temp_im, ( input_resize, input_resize ), mode='constant', preserve_range=True )
    return im

def visualize_map( im, im_mask, pos_map, dst_png, title1, title2, x, y ) :
    fig = plt.figure(figsize=(12.0, 8.0))
    plt.subplot( 1,2,1 )
    plt.title( title1 )
    plt.imshow( im, cm.Greys_r )
    plt.axis( 'off' )
    ax=plt.subplot( 1,2,2 )
    plt.title( title2 )
    plt.imshow( im, cm.Greys_r )
    plt.imshow( pos_map, alpha=0.3, vmin=0.0, vmax=1.0, cmap='jet' )
    circle = patches.Circle((x+16,y+16),radius=16,linewidth=2,edgecolor='red',facecolor='none',alpha=0.3)
    ax.add_patch( circle )
    try :
        plt.contour( im_mask, colors = 'k' )
    except :
        pass
    plt.axis( 'off' )
    plt.subplots_adjust( left = 0.1, right=0.9, top=0.9, bottom=0.1, wspace = 0.1, hspace = 0.1 )
    plt.savefig( dst_png )
    plt.close() 

def calculate_metrics( pos_maps, pos_probs, im_masks ) :
    import ipdb
    ipdb.set_trace()        

    gt_labels = [ 0 if sum( sum( im_mask ) ) == 0 else 1 for im_mask in im_masks ]
    pred_labels = [ 0 if pos_probs < 0.5 else 1 for pos_prob in pos_probs ]
    acc = metrics.accuracy_score( gt_labels, pred_labels )
    fpr, tpr, thresholds = metrics.roc_curve( gt_labels, pos_probs, pos_label=1 )
    auc = metrics.auc( fpr, tpr )
    confusion_mat = metrics.confusion_matrix( gt_labels, pred_labels )
    sensitivity = confusion_mat[1,1] / (confusion_mat[1,1]+confusion_mat[1,0])
    specificity = confusion_mat[0,0] / (confusion_mat[0,0]+confusion_mat[0,1])

    jafroc = 0

    return auc, acc, jafroc, sensitivity, specificity
