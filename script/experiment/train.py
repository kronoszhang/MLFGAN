# -*- coding: utf-8 -*-
"""Train with optional Global Distance, Local Distance, Identification Loss."""
from __future__ import print_function, division



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
#from metric_learn import LMNN

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
#from aligned_reid.dist_metric import DistanceMetric

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1) # for run many time to compute mean result
    parser.add_argument('--set_seed', type=str2bool, default=True) # if True, every time run code, same result would get
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined','generate_market1501'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--resize_h_w', type=eval, default=(256,128))  # (224,224) in paper
    parser.add_argument('--crop_prob', type=float, default=0.4)
    parser.add_argument('--crop_ratio', type=float, default=0.8)  # only `crop_ratio<0` and `crop_prob>0`, crop can be truly run
    parser.add_argument('--ids_per_batch', type=int, default=32)
    parser.add_argument('--ims_per_id', type=int, default=4)# total batch size is : `ids_per_batch` * `ims_per_id`

    # whether write control stdout/stderror to txt file, whether write tensorboardX, whether save model
    parser.add_argument('--log_to_file', type=str2bool, default=True)    
    # whether normalize the  global_features and local_feats
    parser.add_argument('--normalize_feature', type=str2bool, default=True)
    # whether use local feats distance to hard sampler, if False, only use global feats distance to hard sampler 
    # if True and `l_loss_weight`>0, would use `global_feats_dist`,`local_feats_dist`,'global_feats_dist+local_feats_dist' to compute result respectly
    # else only use `global_feats_dist` to compute result with slightly accuracy reduce but save much time 
    parser.add_argument('--local_dist_own_hard_sample',type=str2bool, default=False)    
    
    # loss weight and margin
    parser.add_argument('-gm', '--global_margin', type=float, default=0.3) # margin of global triplet loss
    parser.add_argument('-lm', '--local_margin', type=float, default=0.3)  # margin of local triplet loss
    parser.add_argument('-glw', '--g_loss_weight', type=float, default=1.) # global triplet loss weight in total loss
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.6) # local triplet loss weight in total loss
    parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0.) # id CrossEntropyLoss weight in total loss

    parser.add_argument('--only_test', type=str2bool, default=False) # whether only test without train, if True, a pretrained-model need
    parser.add_argument('--resume', type=bool, default=False) # whether continue train with a special epoch, if True, a pretrained-model need
    parser.add_argument('--exp_dir', type=str, default='')  # where to save model, save tensorboardX result, control stdout/stderror txt file
    parser.add_argument('--model_weight_file', type=str, default='') # not provide pretrained-model but pretrained-model weights file

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp', choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=76)
    parser.add_argument('--staircase_decay_at_epochs',type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',type=float, default=0.1) # decay to 0.1*last_lr
    parser.add_argument('--total_epochs', type=int, default=150) 

    parser.add_argument('--erasing_p', type=float, default=0.7)  # random erasing probability for occlusion
    # how to fuse `base_feats` and `affine_feats` to `feats`, and `feats` would used to compute `global_feats` and `local_feats`
    parser.add_argument('--feat_fuse_strategy', type=str, default='concat',choices=['only_base','concat','max','avg', 'only_affine']) 
    # our model like PAN, should be train with 2 statge, if only `train_base`, this would only use the AlignedReID+reranking (metric learn,
    # PN-GAN) without PAN;if `train_base` first and then `train_all` with the base of `train_base` pretrained model, will lead to fixed
    # base branch and train total network, will lead further improvment, just like PAN
    parser.add_argument('--train_strategy', type=str, default='train_base',choices=['train_base','train_all'])
    parser.add_argument('--norm_base', type=bool, default=False) # whether norm `base_feats`
    parser.add_argument('--norm_affine', type=bool, default=False) # whether norm `affine_feats`
    parser.add_argument('--norm_BaseAffine', type=bool, default=False)  # whether norm `feats` which fused from `base_feats` and `affine_feats`
    # which metric learn to choose, if not `None(euclidean)`, will further improvment with a large margin, but cost more time(about 30 minutes more)
    parser.add_argument('--distance_metric', type=str, default='None',choices=['None','kissme'])

    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 7
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
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = False
    self.train_mirror_type = ['random', 'always', None][0]
    self.train_shuffle = True

    self.test_batch_size = 32  #32 128
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False

    self.erasing_p = args.erasing_p
    self.feat_fuse_strategy = args.feat_fuse_strategy
    self.train_strategy = args.train_strategy
    self.norm_base = args.norm_base
    self.norm_affine = args.norm_affine
    self.norm_BaseAffine = args.norm_BaseAffine
    self.distance_metric = args.distance_metric

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
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      erasing_p=self.erasing_p,
      prng=prng) # only for train set
    self.train_set_kwargs.update(dataset_kwargs) # train set args

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
    self.global_margin = args.global_margin
    self.local_margin = args.local_margin

    # Identification Loss weight
    self.id_loss_weight = args.id_loss_weight
    

    # global loss weight
    self.g_loss_weight = args.g_loss_weight
    # local loss weight
    self.l_loss_weight = args.l_loss_weight

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in a epoch) to log. If only need to log the average information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs(logs dir full name).
    if args.exp_dir == '':
      self.exp_dir = osp.join('exp/train',
        '{}'.format(self.dataset),
        #
        ('nf_' if self.normalize_feature else 'not_nf_') +
        ('ohs_' if self.local_dist_own_hard_sample else 'not_ohs_') +
        'gm_{}_'.format(tfs(self.global_margin)) +
        'lm_{}_'.format(tfs(self.local_margin)) +
        'glw_{}_'.format(tfs(self.g_loss_weight)) +
        'llw_{}_'.format(tfs(self.l_loss_weight)) +
        'idlw_{}_'.format(tfs(self.id_loss_weight)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


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

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)


  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  train_set = create_dataset(**cfg.train_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(feat_fuse_strategy = cfg.feat_fuse_strategy,
                train_strategy = cfg.train_strategy,
                local_conv_out_channels=cfg.local_conv_out_channels,
                num_classes=len(train_set.ids2labels))
  # Model wrapper
  model_w = DataParallel(model)

  #############################
  # Criteria and Optimizers   #
  #############################

  id_criterion = nn.CrossEntropyLoss()
  g_tri_loss = TripletLoss(margin=cfg.global_margin)  # norm branch
  l_tri_loss = TripletLoss(margin=cfg.local_margin)

  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_w.parameters()),  ####  use `model_w.parameters()` or `model.parameters()` is same
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)
  """
  if cfg.train_strategy == 'train_all':
     affine_para1 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para2 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para3 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para4 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para5 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para6 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para7 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para8 = [params for params in model.affine_layer1_Conv.parameters()]
     affine_para9 = [params for params in model.affine_layer1_Conv.parameters()]
 
     affine_para1 = list(map(id,affine_para1))
     affine_para2 = list(map(id,affine_para2))
     affine_para3 = list(map(id,affine_para3))
     affine_para4 = list(map(id,affine_para4))
     affine_para5 = list(map(id,affine_para5))
     affine_para6 = list(map(id,affine_para6))
     affine_para7 = list(map(id,affine_para7))
     affine_para8 = list(map(id,affine_para8))
     affine_para9 = list(map(id,affine_para9))

     base_params = filter(lambda p:id(p) not in affine_para1+affine_para2+affine_para3+affine_para4+affine_para5+affine_para6+affine_para7
                                                +affine_para8+affine_para9, filter(lambda p: p.requires_grad, model.parameters()))
     optimizer = optim.Adam([{'params':base_params},
                             {'params':affine_para1, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para2, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para3, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para4, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para5, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para6, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para7, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para8, 'lr': cfg.base_lr*0.001},
                             {'params':affine_para9, 'lr': cfg.base_lr*0.001},],
                             lr=cfg.base_lr,
                             weight_decay=cfg.weight_decay)  # try this then"""  # this would lead error

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  

  ################################
  # May Resume Models and Optims #
  ################################  

  if cfg.resume or (cfg.train_strategy=='train_all'):
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  ########
  # Test #
  ########

  def test(train_strategy, norm_base, norm_affine, norm_BaseAffine, feat_fuse_strategy,load_model_weight=False):
    """this func is used for test model and get accuracy(mAP,CMC) result"""
    # whether load weight file of total model directly in a pointed path
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)
 
    # to get `train_set's global_feats` to train `metric`
    assert cfg.distance_metric in ['None','kissme']
    if cfg.distance_metric != 'None':
      global_metric = DistanceMetric(algorithm=cfg.distance_metric)
      local_metric = DistanceMetric(algorithm=cfg.distance_metric)
      global_feats, local_feats, ids = [], [], []
      epoch_done = False
      step = 0
      printed = False
      st = time.time()
      last_time = time.time()
      print('\n=========> Use train set to train metric <=========\n')
      print('Extracting Features ...')
      while not epoch_done:
        ims, im_names, labels, mirrored, epoch_done = train_set.next_batch() 
        old_train_eval_model = model_w.training # check whether in train, if now in train, will return True, else return False
        model_w.eval()
        ims = Variable(TVT(torch.from_numpy(ims).float()))
        # use model to get `batch train_set` `global_feat` and `local_feats`
        global_feat, local_feat = model_w(ims, train_strategy, norm_base, norm_affine, norm_BaseAffine, feat_fuse_strategy)[:2]
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        model_w.train(old_train_eval_model)
        global_feats.append(global_feat) # get all `train_set global_feats` 
        local_feats.append(local_feat) # get all `train_set local_feats` 
        ids.append(labels) # get all `train_set labels` 
        # log
        total_batches = (train_set.prefetcher.dataset_size // train_set.prefetcher.batch_size + 1)
        step += 1
        if step % 2 == 0:
          if not printed:
            printed = True
          else:
            # Clean the current line
            sys.stdout.write("\033[F\033[K")
          print('{}/{} batches done, +{:.2f}s, total {:.2f}s'.format(step, total_batches, time.time() - last_time, time.time() - st))  
          last_time = time.time()
        
      print('\nUsing Features to train metric ...')
      global_feats = np.vstack(global_feats)
      # to concate one images all parts' local features to form a full part features, 
      # other fusion strategy can be used in here,like max or avg, weighted diff parts, we obey AlignedReid paper here
      local_feats = np.concatenate(local_feats)  
      ids = np.hstack(ids)
         

      # we only use `global_feats` and `ids` to train `metric`, if we use `local_feats`, would get further accuracy improvment, but will pay
      # many time, and the following `local_feats metric` leads some numerical bug (about `raise LinAlgError("Singular matrix")`), we can address
      # this problem in the future,
      # Note: the comment here are matched with `TestSe.TestSet.eval` code after `global_feats = global_metric.transform(global_feats) ` 
      # use `train_set's global_feats` to train `global_metric(transfer test_set's global_feats to metric global feat, improve result)`
      global_metric.train(global_feats, ids) 
      # use `train_set's local_feats` to train `local_metric(transfer test_set's local_feats to metric local feat, improve result)` 
      #N, H, C = local_feats.shape
      #local_feats = local_feats.reshape(N,-1)
      #local_metric.train(local_feats, ids) 
      #global_metric = LMNN(k=3)#################################################################used for other metric
      #global_metric.fit(global_feats, ids) #################################################################used for other metric
    else: # choose `None` for `metric_distance_learn`, we would not use metric learn to save time
      global_metric, local_metric = 'None', 'None'
 
    use_local_distance = (cfg.l_loss_weight > 0) and cfg.local_dist_own_hard_sample  # decide whether use local distance in  test stage

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      # call `TestSet.evel to test model`
      test_set.eval(
        global_metric = global_metric,
        local_metric = local_metric,
        distance_metric = cfg.distance_metric,
        train_strategy=cfg.train_strategy, 
        feat_fuse_strategy=cfg.feat_fuse_strategy, 
        norm_base=cfg.norm_base, 
        norm_affine=cfg.norm_affine, 
        norm_BaseAffine=cfg.norm_BaseAffine,
        normalize_feat=cfg.normalize_feature,
        use_local_distance=use_local_distance)

  
  if cfg.only_test:
    test(cfg.train_strategy, cfg.norm_base, cfg.norm_affine, cfg.norm_BaseAffine, cfg.feat_fuse_strategy,load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    g_prec_meter = AverageMeter()  # use global feature to compute accuracy without margin
    g_m_meter = AverageMeter() # use global feature to compute accuracy with margin
    g_dist_ap_meter = AverageMeter()
    g_dist_an_meter = AverageMeter()
    g_loss_meter = AverageMeter()

    l_prec_meter = AverageMeter()
    l_m_meter = AverageMeter()
    l_dist_ap_meter = AverageMeter()
    l_dist_an_meter = AverageMeter()
    l_loss_meter = AverageMeter()

    id_loss_meter = AverageMeter()

    loss_meter = AverageMeter()  # total loss

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()  # get a batch data from train_set
      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      labels_var = Variable(labels_t)

      # use model to forward and get return
      global_feat, local_feat, logits = model_w(ims_var, train_strategy = cfg.train_strategy, feat_fuse_strategy=cfg.feat_fuse_strategy,
                                                norm_base=cfg.norm_base, norm_affine=cfg.norm_affine, norm_BaseAffine=cfg.norm_BaseAffine)  
      # use `global_feat, local_feat` to compute ranking loss(triplet loss)
      # global loss
      g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(g_tri_loss, global_feat, labels_t,normalize_feature=cfg.normalize_feature)
      # local loss
      if cfg.l_loss_weight == 0:
        l_loss = 0
      elif cfg.local_dist_own_hard_sample:
        # Let local distance find its own hard samples.
        l_loss, l_dist_ap, l_dist_an, _ = local_loss(l_tri_loss, local_feat, None, None, labels_t,normalize_feature=cfg.normalize_feature)
      else:
        # use global feature to hard samples
        l_loss, l_dist_ap, l_dist_an = local_loss(l_tri_loss, local_feat, p_inds, n_inds, labels_t,normalize_feature=cfg.normalize_feature)

      # use `logits` to compute classification loss(softmax classification, CrossEntropyLoss)
      id_loss = 0
      if cfg.id_loss_weight > 0:
        id_loss = id_criterion(logits, labels_var)


      # get total loss
      loss = g_loss * cfg.g_loss_weight + l_loss * cfg.l_loss_weight + id_loss * cfg.id_loss_weight 
        
        
      # backword
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()




      ############
      # Step Log #
      ############
   
      # precision in train
      g_prec = (g_dist_an > g_dist_ap).data.float().mean()
      # the proportion of triplets that satisfy margin
      g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
      g_d_ap = g_dist_ap.data.mean()
      g_d_an = g_dist_an.data.mean()

      g_prec_meter.update(g_prec)
      g_m_meter.update(g_m)
      g_dist_ap_meter.update(g_d_ap)
      g_dist_an_meter.update(g_d_an)
      g_loss_meter.update(to_scalar(g_loss))

      if cfg.l_loss_weight > 0:
        # precision in train
        l_prec = (l_dist_an > l_dist_ap).data.float().mean()
        # the proportion of triplets that satisfy margin
        l_m = (l_dist_an > l_dist_ap + cfg.local_margin).data.float().mean()
        l_d_ap = l_dist_ap.data.mean()
        l_d_an = l_dist_an.data.mean()

        l_prec_meter.update(l_prec)
        l_m_meter.update(l_m)
        l_dist_ap_meter.update(l_d_ap)
        l_dist_an_meter.update(l_d_an)
        l_loss_meter.update(to_scalar(l_loss))


      if cfg.id_loss_weight > 0:
        id_loss_meter.update(to_scalar(id_loss))

      loss_meter.update(to_scalar(loss))

      if step % cfg.log_steps == 0:
        time_log = '\t{} Step {}/Ep {}, {:.2f}s'.format(
          cfg.train_strategy, train_step, ep + 1, time.time() - step_st, )

        # global log
        if cfg.g_loss_weight > 0:
          g_log = (', gp {:.2%}, gm {:.2%}, '
                   'gd_ap {:.4f}, gd_an {:.4f}, '
                   'gL {:.4f}'.format(
            g_prec_meter.val, g_m_meter.val,
            g_dist_ap_meter.val, g_dist_an_meter.val,
            g_loss_meter.val, ))
        else:
          g_log = ''
    
        # local log
        if cfg.l_loss_weight > 0:
          l_log = (', lp {:.2%}, lm {:.2%}, '
                   'ld_ap {:.4f}, ld_an {:.4f}, '
                   'lL {:.4f}'.format(
            l_prec_meter.val, l_m_meter.val,
            l_dist_ap_meter.val, l_dist_an_meter.val,
            l_loss_meter.val, ))
        else:
          l_log = ''
     
        # id loss log
        if cfg.id_loss_weight > 0:
          id_log = (', idL {:.4f}'.format(id_loss_meter.val))
        else:
          id_log = ''

        # total loss log
        total_loss_log = ', loss {:.4f}'.format(loss_meter.val)

        # print all log
        log = time_log + g_log + l_log + id_log + total_loss_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = '{}, Ep {}, {:.2f}s'.format(cfg.train_strategy, ep + 1, time.time() - ep_st, )

    # global log
    if cfg.g_loss_weight > 0:
      g_log = (', gp {:.2%}, gm {:.2%}, '
               'gd_ap {:.4f}, gd_an {:.4f}, '
               'gL {:.4f}'.format(
        g_prec_meter.avg, g_m_meter.avg,
        g_dist_ap_meter.avg, g_dist_an_meter.avg,
        g_loss_meter.avg, ))
    else:
      g_log = ''

    # local log
    if cfg.l_loss_weight > 0:
      l_log = (', lp {:.2%}, lm {:.2%}, '
               'ld_ap {:.4f}, ld_an {:.4f}, '
               'lL {:.4f}'.format(
        l_prec_meter.avg, l_m_meter.avg,
        l_dist_ap_meter.avg, l_dist_an_meter.avg,
        l_loss_meter.avg, ))
    else:
      l_log = ''

    # id loss log
    if cfg.id_loss_weight > 0:
      id_log = (', idL {:.4f}'.format(id_loss_meter.avg))
    else:
      id_log = ''

    # total loss log
    total_loss_log = ', loss {:.4f}'.format(loss_meter.avg)

    # print all log
    log = time_log + g_log + l_log + id_log + total_loss_log
    print(log)


    # Log to TensorBoard
    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'loss',
        dict(global_loss=g_loss_meter.avg,
             local_loss=l_loss_meter.avg,
             id_loss=id_loss_meter.avg,
             loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'tri_precision',
        dict(global_precision=g_prec_meter.avg,
             local_precision=l_prec_meter.avg,),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(global_satisfy_margin=g_m_meter.avg,
             local_satisfy_margin=l_m_meter.avg,),
        ep)
      writer.add_scalars(
        'global_dist',
        dict(global_dist_ap=g_dist_ap_meter.avg,
             global_dist_an=g_dist_an_meter.avg,),
        ep)
      writer.add_scalars(
        'local_dist',
        dict(local_dist_ap=l_dist_ap_meter.avg,
             local_dist_an=l_dist_an_meter.avg,),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########
  # test if train is finished
  test(cfg.train_strategy, cfg.norm_base, cfg.norm_affine, cfg.norm_BaseAffine, cfg.feat_fuse_strategy,load_model_weight=False)


if __name__ == '__main__':
  main()
