from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
from metric_learn import LMNN
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')
from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase
from aligned_reid.dist_metric import DistanceMetric

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(1,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False) 
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--resize_h_w', type=eval, default=(256,128)) 

    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--local_dist_own_hard_sample',type=str2bool, default=False)    
    
  
    parser.add_argument('--exp_dir', type=str, default='./exp/500_151_150_75')  # model path

    parser.add_argument('--test_batch_size', type=int, default=32)

    parser.add_argument('--feat_fuse_strategy', type=str, default='concat',choices=['only_base','concat','max','avg', 'only_affine']) 
    parser.add_argument('--train_strategy', type=str, default='train_base',choices=['train_base','train_all'])
    parser.add_argument('--norm_base', type=bool, default=False) 
    parser.add_argument('--norm_affine', type=bool, default=False) 
    parser.add_argument('--norm_BaseAffine', type=bool, default=False)  
    parser.add_argument('--query_image_name', type=str, default='00000120_0006_00000016.jpg')    # 00000001_0001_00000000.jpg  00000015_0001_00000002.jpg  00000008_0006_00000002.jpg  00001501_0006_00000004.jpg  00000684_0004_00000006.jpg  00000008_0001_00000008.jpg  00000066_0002_00000004.jpg  00000120_0006_00000016.jpg
 
    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged. `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to exactly reproduce the result in training, you have to set num of threads to 1.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]



    self.test_batch_size = args.test_batch_size
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False

    self.feat_fuse_strategy = args.feat_fuse_strategy
    self.train_strategy = args.train_strategy
    self.norm_base = args.norm_base
    self.norm_affine = args.norm_affine
    self.norm_BaseAffine = args.norm_BaseAffine
    self.query_image_name = args.query_image_name

    # for both train set and test set
    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)# only for test set
    self.test_set_kwargs.update(dataset_kwargs) # test set args

    ###############
    # ReID Model  #
    ###############

    self.local_dist_own_hard_sample = args.local_dist_own_hard_sample

    self.normalize_feature = args.normalize_feature

    self.local_conv_out_channels = 128

    # The root dir of logs(logs dir full name).
    self.exp_dir = args.exp_dir
    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims, train_strategy, feat_fuse_strategy, norm_base, norm_affine, norm_BaseAffine):
    old_train_eval_model = self.model.training  # check whether in train, if now in train, will return True, else return False
    # Set eval mode.  Force all BN layers to use global mean and variance, also disable dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    # use model to extract `global_feats` and `local_feats` of `test_set`
    global_feat, local_feat = self.model(ims, train_strategy, norm_base, norm_affine, norm_BaseAffine, feat_fuse_strategy)[:2] 
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat  # type : numpy.array


def main():
  cfg = Config()

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)


  ###########
  # Dataset #
  ###########

  test_sets = create_dataset(**cfg.test_set_kwargs)
  test_set_names = cfg.dataset

  ###########
  # Models  #
  ###########
  if test_set_names == 'market1501':
    num_classes = 750 + 1 # `1` is for `0000` id
  elif test_set_names == 'cuhk03':
    num_classes = 700
  elif test_set_names == 'duke':
    num_classes = 702

  model = Model(feat_fuse_strategy = cfg.feat_fuse_strategy,
                train_strategy = cfg.train_strategy,
                local_conv_out_channels=cfg.local_conv_out_channels,
                num_classes=num_classes)
  # Model wrapper
  model_w = DataParallel(model)
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=2e-4,
                         weight_decay=0.0005) # `optimizer` is used for train, but useless in here, only for load model

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  ########
  # Test #
  ########

  def test(query_image_name,dataset,train_strategy, norm_base, norm_affine, norm_BaseAffine, feat_fuse_strategy,load_model_weight=False):
    """this func is used for test model and get accuracy(mAP,CMC) result"""
    # whether load weight file of total model directly in a pointed path
    if load_model_weight:
      load_ckpt(modules_optims, cfg.ckpt_file)
 
    use_local_distance = cfg.local_dist_own_hard_sample  

   
    test_sets.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n=========> Test on dataset: {} <=========\n'.format(test_set_names))
    distmap,q_im_name, g_im_name,mq_im_name,q_im_id,g_im_id,mq_im_id,q_im_cam,g_im_cam,mq_im_cam = test_sets.demo(
        train_strategy=cfg.train_strategy, 
        feat_fuse_strategy=cfg.feat_fuse_strategy, 
        norm_base=cfg.norm_base, 
        norm_affine=cfg.norm_affine, 
        norm_BaseAffine=cfg.norm_BaseAffine,
        normalize_feat=cfg.normalize_feature,
        use_local_distance=use_local_distance)
    if len(distmap) == 6: # use global dist and local dist and rerank
      dist = distmap[5]
    elif len(distmap) == 3: # use global dist and local dist and no rerank
      dist = distmap[2]
    elif len(distmap) == 2: # only use global dist and rerank
      dist = distmap[1]
    else: # len(distmap) == 1  only use global dist and no rerank
      dist = distmap[0]
    


    # sort images and return rank list  
    query_id, query_cam = int(query_image_name.split('_')[0]), int(query_image_name.split('_')[1])
    index = sort_img(dist, query_id, query_cam, g_im_id, g_im_cam)

    print(g_im_name)
    # show
    query_path = osp.join('../../Dataset', dataset, 'images', query_image_name)
    print("query image is:",query_image_name)
    print('Top 10 images are as follow:')
    # Visualize Ranking Result 
    try: 
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        ids = []
        print("matched gallery images are:")
        for i in range(10):  # show top-10
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path = g_im_name[index[i]]
            ids.append(g_im_id[index[i]])  
            img_path = osp.join('../../Dataset', dataset, 'images', img_path)
            
            if ids[-1] == query_id:
                color = 'green'
            else:
                color = 'red'
            # label whether demo right `image.size=(128,64)`
            plt.gca().add_patch(plt.Rectangle(xy=(0, 0),
                                              width=62, 
                                              height=126,
                                              edgecolor= color,
                                              fill=False, linewidth=2))
            imshow(img_path,'%d'%(i+1))
            print(img_path)
    except RuntimeError:
        for i in range(10): 
            img_path = g_im_name[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    print(query_id, ids)
    fig.savefig("demo.png") 
   

  ########
  # Test #
  ########
  test(cfg.query_image_name,
       cfg.dataset,
       cfg.train_strategy, 
       cfg.norm_base, 
       cfg.norm_affine, 
       cfg.norm_BaseAffine, 
       cfg.feat_fuse_strategy,
       load_model_weight=True)

#######################################################################
# sort the images

def sort_img(distmat, ql, qc, gl, gc):    
        
    # predict index
    index = np.argsort(distmat)  #from small to large
    #index = index[::-1]  # if score, use this, if dist, comment this
    
    # good index
    query_index = np.argwhere(gl==ql)
    # same camera
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)  
    junk_index = np.append(junk_index2, junk_index1) 

   
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True) 
    q_num, g_num = index.shape 
    mask = mask.reshape(q_num, g_num)
    index = index[mask]  
    
    return index # good index
                 
#####################################################################
#Show result

def imshow(path, title=None):
    """imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  


if __name__ == '__main__':
  main()



