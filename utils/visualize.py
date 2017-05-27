import os

import numpy as np
from skimage import transform, io
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def imread(image_path, resize=500):

    im = io.imread(image_path)
    o_h, o_w = im.shape
    
    resize_ratio = resize/float(max(o_h,o_w))
    r_h, r_w = int(o_h*resize_ratio), int(o_w*resize_ratio)

    pad_h1 = (resize-r_h)//2
    pad_h2 = resize-r_h-pad_h1
    pad_w1 = (resize-r_w)//2
    pad_w2 = resize-r_w-pad_w1

    resized = transform.resize(im, (r_h,r_w))
    padded = np.pad(resized, ((pad_h1,pad_h2),(pad_w1,pad_w2)),mode='constant')

    return padded


def vis_pos(image_path, image, feature_map, label, prob, save_dir=None, other_src_image=None, note=None):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_dim = image.shape[0]

    resized_activations = transform.resize(feature_map, (input_dim,input_dim),preserve_range=True)
    
    # centering activations
    centered_activations = resized_activations - np.min(resized_activations)

    # normalize activation values
    norm_activations = centered_activations / (np.max(resized_activations)-np.min(resized_activations))
    norm_activations = norm_activations * prob

    # parsing image file name from image_path
    img_fname = image_path.split('/')[-1]
    
    if label == 1:
        img_disease = image_path.split('/')[4]
        save_dir_abnormal = os.path.join(save_dir, 'abnormal')
        if not os.path.exists(save_dir_abnormal):
            os.makedirs(save_dir_abnormal)

        save_dst = os.path.join(save_dir_abnormal,img_fname)
    else:
        img_disease = 'normal'
        save_dir_normal = os.path.join(save_dir, 'normal')
        if not os.path.exists(save_dir_normal):
            os.makedirs(save_dir_normal)

        save_dst = os.path.join(save_dir_normal,img_fname)

    dpi = 96.0
    xinch = 1400/dpi
    yinch = 800/dpi
    plt.figure(figsize=(xinch,yinch))
    #ax = plt.axes([0,0,1,1],frame_on=False,xticks=[],yticks=[])


    #plt.figure()
    plt.subplot(1,2,1)
    if note is None:
        plt.title('%s'%img_disease)
    else:
        title_text = '%s\n%s' % (img_disease, note)
        plt.title(title_text)
    
    if other_src_image is None:
        plt.imshow(image, cm.Greys_r)
    else:
        plt.imshow(other_src_image, cm.Greys_r)

    plt.subplot(1,2,2)
    plt.imshow(image,cm.Greys_r)
    plt.imshow(norm_activations, alpha=0.3, vmin=0.0, vmax=1.0)
    plt.title('Abnormal Prob.: %.4f'%prob)
    plt.savefig(save_dst)
    plt.close()

