from easydict import EasyDict as edict
import numpy as np

"""
1 run `openpose.exe in windows command or run openpose.bin in Linux terimal and obey OpenPose GitHub order to generate images pose and save pose images`
2 run `clustering.py` to cluster all generated pose images to 8 classes
3 run `train.py` to train GAN and Discrimator Network
4 run `evaluate.py` to test model with arbitrary poses in `sample poses folder`
5 run `generate.py` to use model to generate 8 images with 8 cannonical poses in `cannonical_poses folder` respectly and copy raw imagesfor latter reid
"""
__C = edict()
cfg = __C

# Other options
__C.NET = 'GAN'
__C.GPU_ID = '0' # use GPU 0 to train
__C.NUM_CLASS = 751
__C.DEBUG = False
__C.FILE_PATH = './'   # working directory

# Train options
__C.TRAIN = edict()
__C.TRAIN.imgs_path = '../../Market-1501-v15.09.15/bounding_box_train'
__C.TRAIN.pose_path = '../../poses' 
__C.TRAIN.idx_path = '../train_idx.txt'
__C.TRAIN.LR = 0.0002
__C.TRAIN.LR_DECAY = 10
__C.TRAIN.MAX_EPOCH = 20

__C.TRAIN.DISPLAY = 100
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.NUM_WORKERS = 32

# GAN Network or Loss options
__C.TRAIN.ngf = 64
__C.TRAIN.ndf = 64
__C.TRAIN.num_resblock = 9
__C.TRAIN.lambda_idt = 10
__C.TRAIN.lambda_att = 1


# Test options
__C.TEST = edict()
__C.TEST_GENERATE = True # `True` for generate and `False` for evaluate when run `evaluate.py` or can directly run `generate.py`
#__C.TEST.imgs_path = '../../Market-1501-v15.09.15/bounding_box_train'  # to generate images for `bounding_box_train`
#__C.TEST.imgs_path = '../../Market-1501-v15.09.15/bounding_box_test'   # to generate images for `bounding_box_test`
#__C.TEST.imgs_path = '../../Market-1501-v15.09.15/query'               # to generate images for `query`
__C.TEST.imgs_path = '../../Market-1501-v15.09.15/gt_bbox'              # to generate images for `gt_bbox`
#__C.TEST.imgs_path = '../../Market-1501-v15.09.15/bounding_box_test'   # to evaluate model with `bounding_box_test`
__C.TEST.pose_path = '../../can_poses'   # to generate with 8 poses  cannonical_poses
#__C.TEST.pose_path = '../../sample_pose'         # to evaluate model, use `sample_pose folder` which contain arbitrary poses
__C.TEST.idx_path = '../test_idx.txt'       # didn't use this in program
__C.TEST.BATCH_SIZE = 1                     # this must be 1
__C.TEST.GPU_ID = '0'

# Generate images for latter reid # choose from '../../sample poses'  and '../../cannonical_poses' 
__C.GENERATE = edict()
__C.GENERATE.imgs_path = './Market-1501-v15.09.15'  # generate all images in Market1501
__C.GENERATE.pose_path = './cannonical_poses'   # to generate images with 8 special poses in `cannonical_poses folder` for all dataset images
__C.GENERATE.BATCH_SIZE = 1  # this must be 1
__C.GENERATE.GPU_ID = '0'
