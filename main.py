import shutil, os, sys
sys.path.append( '/lunit/home/sgpark/projects/libs/lunit-handler/' )
from lunithandler import lflags
import cv2
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
tf.logging.set_verbosity(tf.logging.ERROR)

TRIAL = '170529_base_triplet'
MODEL = 'alexnet'

def load_trn_options():

    # environments
    lflags.DEFINE_string('trial', TRIAL, """Trial id.""")
    lflags.DEFINE_string('train_csv', 'inputs/looksearch_trn.csv', """CSV containing data info for training""")
    lflags.DEFINE_string('train_dir', '/lunit/home/sgpark/archive/looksearch/%s'%TRIAL,
                         """Directory where to write event logs and checkpoint.""")
    lflags.DEFINE_string('model', 'models.%s'%MODEL, """Model to be imported.""")
    lflags.DEFINE_list('gpu_ids', [0,1,2,3], """ID of gpus to use.""")
    lflags.DEFINE_list('num_classes', [17,20], """Number of classes.""")

    # Preprocessing parameters
    lflags.DEFINE_integer('input_size', 256, """Target image size before random cropping.""")
    lflags.DEFINE_integer('crop_size', 227, """Image size feeding to network.""")
    lflags.DEFINE_integer('channel_mean', 100, """Channel mean value.""")
    lflags.DEFINE_bool('reverse_grayscale', False, """Randomly reverse grayscale""")
    lflags.DEFINE_bool('horizontal_flip', False, """Randomly flip image horizontally""")
    lflags.DEFINE_bool('rotation', False, """Randomly rotate image""")
    lflags.DEFINE_list('brightness', [False, 0.1], """Randomly adjust brightness""")
    lflags.DEFINE_list('gamma', [False, 0.25], """Randomly adjust brightness""")
    lflags.DEFINE_list('contrast', [False, 0.7, 1.3], """Randomly adjust contrast""")
    #lflags.DEFINE_list('blur', [True, 0.5, 1.5], """Randomly blur""")
    #lflags.DEFINE_list('sharpen', [True, 0.5, 1.5, 0.5, 1.5], """Randomly sharpen""")
    lflags.DEFINE_list('blur_or_sharpen', [False, 0.5, 1.5, 0.5, 1.5], """Randomly blur or sharpen""")

    # Optimizer
    # NOTE: currently support 'rmsprop' and 'sgd_with_momentum' only
    # in case of 'sgd_with_momentum'
    lflags.DEFINE_string('optimizer', 'sgd_with_momentum', """Optimizer for training.""")
    lflags.DEFINE_float('momentum', 0.9, """Momentum parameter for RMSPROP.""")
    lflags.DEFINE_bool('use_nesterov', True, """Nesterov momentum""")

    # in case of 'rmsprop'
    #lflags.DEFINE_string('optimizer', 'rmsprop', """Optimizer for training.""")
    #lflags.DEFINE_float('decay', 0.9, """Decay parameter for RMSPROP.""")
    #lflags.DEFINE_float('momentum', 0.9, """Momentum parameter for RMSPROP.""")
    #lflags.DEFINE_float('epsilon', 0.1, """Epsilon parameter for RMSPROP.""")

    # training parameters
    lflags.DEFINE_integer('save_step', 1, """ """)
    lflags.DEFINE_integer('batch_size', 256, """""")
    lflags.DEFINE_list('epoch_schedule', [2, 1, 1], 
    #lflags.DEFINE_list('epoch_schedule', [10, 20, 30], 
                       """Learning epochs schedule. Note that the end of list means total number of epochs.""")
    lflags.DEFINE_list('lr_schedule', [0.001, 0.0001, 0.00001], """Learning rate schedule.""")
    lflags.DEFINE_list('alpha_schedule', [0.5, 0.5, 0.5], """Alpha schedule, weight for the classification loss""")
    lflags.DEFINE_float('weight_decay', 0.0001, """Weight decay parameter.""")

def load_val_options():

    lflags.DEFINE_string('val_csv',
                         'inputs/ip_base_tst.csv',
                         """CSV containing data info for training""")
    lflags.DEFINE_integer('batch_size', 1, """Number of images to process in a batch.""", allow_overwrite=True)
    lflags.DEFINE_integer('input_size', 700, """Target image size before random cropping.""", allow_overwrite=True)
    lflags.DEFINE_integer('crop_size', 700, """Image size feeding to network.""", allow_overwrite=True)
    lflags.DEFINE_integer('gpu_id', 0, """Image size feeding to network.""", allow_overwrite=True)

def main(argv=None):

    if argv[1] == 'trn':
        # load options for training
        load_trn_options()
        
        if not os.path.exists(lflags.train_dir):
            os.makedirs(lflags.train_dir)
        shutil.copy(__file__,os.path.join(lflags.train_dir,argv[0]))
        import finetune
        finetune.train()

    elif argv[1] == 'val':
        # load options for validation
        # NOTE: options for training should be called first.
        #       As they have several options to be used for validation
        load_trn_options()
        load_val_options()
        
        if not os.path.exists(lflags.train_dir):
            print 'train_dir does not exist!'
            return False

        import validate
        validate.validate()

if __name__ == '__main__':
    tf.app.run()

