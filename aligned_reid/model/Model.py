import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .resnet import resnet50, Bottleneck
import sys
sys.path.insert(0, '..')
from ..utils.distance import normalize

class Model(nn.Module):
  def __init__(self, feat_fuse_strategy, train_strategy, local_conv_out_channels=128, num_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    self.inplanes = 512

    assert train_strategy in ['train_base', 'train_all']
    if train_strategy == 'train_base':
      planes = 2048
    else:
      assert feat_fuse_strategy in ['only_base','concat', 'max', 'avg', 'only_affine']  
      if feat_fuse_strategy == 'concat':
        planes = 4096
      else:
        planes = 2048
    
    if train_strategy == 'train_all':  # if `train_base`, not use follow layers
      for p in self.parameters(): # if fine tune all, we mask and fixed base branch
        p.requires_grad = False
      
      # affine network layers : a res block and a avg pool
      # res block
      self.affine_layer1_Conv = nn.Conv2d(512*4, 512*8, kernel_size=1, stride=1, padding=0, bias=False)
      self.affine_layer1_BN = nn.BatchNorm2d(512*8)
      self.affine_layer1_Relu = nn.ReLU(inplace=True)
      self.affine_layer2_Conv = nn.Conv2d(512*8, 512*4, kernel_size=1, stride=1, padding=0, bias=False)
      self.affine_layer2_BN = nn.BatchNorm2d(512*4)
      self.affine_layer2_Relu = nn.ReLU(inplace=True)
      # avg pool
      self.affine_layer3_AvgPool = nn.AdaptiveAvgPool2d((1, 1))
      # FCN to change channel
      self.affine_layer3_Conv = nn.Conv2d(512*4, 6, kernel_size=1, stride=1, padding=0, bias=False)
      self.affine_layer3_BN = nn.BatchNorm2d(6)
      # for those layers' init
      init.kaiming_normal_(self.affine_layer1_Conv.weight.data, a=0, mode= 'fan_in')
      init.normal_(self.affine_layer1_BN.weight.data, 1.0, 0.02)
      init.constant_(self.affine_layer1_BN.bias.data, 0.0)
      init.kaiming_normal_(self.affine_layer2_Conv.weight.data, a=0, mode= 'fan_in')
      init.normal_(self.affine_layer2_BN.weight.data, 1.0, 0.02)
      init.constant_(self.affine_layer2_BN.bias.data, 0.0)
      init.kaiming_normal_(self.affine_layer3_Conv.weight.data, a=0, mode= 'fan_in')
      init.normal_(self.affine_layer3_BN.weight.data, 1.0, 0.02)
      init.constant_(self.affine_layer3_BN.bias.data, 0.0)
       

      # affine branch, we default `resnet50`. if others network, `block` and `layers` would be different, you can see this in `resnet.py`
      block = Bottleneck    # resnet50
      layers = [3, 4, 6, 3] # resnet50
      self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # we use the same struct as base layer3, but not share param and init differently
      self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
      # to init `layer3` and 'layer4'
      init.kaiming_normal_(self.layer3[0].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[0].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[0].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[1].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[1].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[1].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[2].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[2].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[2].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[3].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[3].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[3].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[4].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[4].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[4].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[5].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[5].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer3[5].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[0].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[0].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[0].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[1].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[1].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[1].conv3.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[2].conv1.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[2].conv2.weight.data, a=0, mode= 'fan_in')
      init.kaiming_normal_(self.layer4[2].conv3.weight.data, a=0, mode= 'fan_in')

      init.normal_(self.layer3[0].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[0].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[0].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[1].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[1].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[1].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[2].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[2].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[2].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[3].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[3].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[3].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[4].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[4].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[4].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[5].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[5].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer3[5].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[0].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[0].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[0].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[1].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[1].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[1].bn3.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[2].bn1.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[2].bn2.weight.data, 1.0, 0.02)
      init.normal_(self.layer4[2].bn3.weight.data, 1.0, 0.02)

      init.constant_(self.layer3[0].bn1.bias.data, 0.0)
      init.constant_(self.layer3[0].bn2.bias.data, 0.0)
      init.constant_(self.layer3[0].bn3.bias.data, 0.0)
      init.constant_(self.layer3[1].bn1.bias.data, 0.0)
      init.constant_(self.layer3[1].bn2.bias.data, 0.0)
      init.constant_(self.layer3[1].bn3.bias.data, 0.0)
      init.constant_(self.layer3[2].bn1.bias.data, 0.0)
      init.constant_(self.layer3[2].bn2.bias.data, 0.0)
      init.constant_(self.layer3[2].bn3.bias.data, 0.0)
      init.constant_(self.layer3[3].bn1.bias.data, 0.0)
      init.constant_(self.layer3[3].bn2.bias.data, 0.0)
      init.constant_(self.layer3[3].bn3.bias.data, 0.0)
      init.constant_(self.layer3[4].bn1.bias.data, 0.0)
      init.constant_(self.layer3[4].bn2.bias.data, 0.0)
      init.constant_(self.layer3[4].bn3.bias.data, 0.0)
      init.constant_(self.layer3[5].bn1.bias.data, 0.0)
      init.constant_(self.layer3[5].bn2.bias.data, 0.0)
      init.constant_(self.layer3[5].bn3.bias.data, 0.0)
      init.constant_(self.layer4[0].bn1.bias.data, 0.0)
      init.constant_(self.layer4[0].bn2.bias.data, 0.0)
      init.constant_(self.layer4[0].bn3.bias.data, 0.0)
      init.constant_(self.layer4[1].bn1.bias.data, 0.0)
      init.constant_(self.layer4[1].bn2.bias.data, 0.0)
      init.constant_(self.layer4[1].bn3.bias.data, 0.0)
      init.constant_(self.layer4[2].bn1.bias.data, 0.0)
      init.constant_(self.layer4[2].bn2.bias.data, 0.0)
      init.constant_(self.layer4[2].bn3.bias.data, 0.0)
    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal_(self.fc.weight, std=0.001)
      init.constant_(self.fc.bias, 0)

    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)
    
  def forward(self, x, train_strategy, norm_base, norm_affine, norm_BaseAffine, feat_fuse_strategy):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    ### norm branch
    # shape [N, C, H, W]
    feat, block2 = self.base(x)
  
    assert train_strategy in ['train_base', 'train_all']
    if train_strategy == 'train_base':
      # only use base branch when fuse
      # now use fused feat to run aligned reid
      global_feat = F.avg_pool2d(feat, feat.size()[2:])
      # shape [N, C]
      global_feat = global_feat.view(global_feat.size(0), -1)
      # shape [N, C, H, 1]
      local_feat = torch.mean(feat, -1, keepdim=True)
      local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
      # shape [N, H, c]
      local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
      if hasattr(self, 'fc'):
        logits = self.fc(global_feat)
        return global_feat, local_feat, logits
      return global_feat, local_feat
    if train_strategy == 'train_all':
      # 1. compute base branch and affine branch 
      # (1). base branch
      base_feat = feat # shape [N, C, H, W]
      # (2).use `block4 result, i.e. feat` to conduct `block2` affine transformation
      # first: let `feat` go through a layer `ResBlock` of Grid Network
      affine = self.affine_layer1_Conv(feat)
      affine = self.affine_layer1_BN(affine) 
      affine = self.affine_layer1_Relu(affine)
      affine = self.affine_layer2_Conv(affine)
      affine = self.affine_layer2_BN(affine)
      affine = affine + feat
      # second: let `affine` go through a layer `AvgPool` of Grid Network
      affine = self.affine_layer3_AvgPool(affine) # shape: N*1*1*(512*4)
      affine = self.affine_layer3_Conv(affine)
      affine = self.affine_layer3_BN(affine)
      affine = affine.float()
      # third: reshape this to affine para
      bz = affine.shape[0]
      affine = affine.reshape(bz, 2, 3)
      # forth: use this to run affine transform in `block2`
      grid = F.affine_grid(affine, block2.size())
      output = F.grid_sample(block2, grid)
      affine_result = output
      # fifth: go through aligned branch
      affine_result = affine_result.cuda()
      affine_result = self.layer3(affine_result)
      affine_result = self.layer4(affine_result)
      # sixth: get affine branch feature
      affine_feat = affine_result 
      # 2. use base branch and affine branch to fuse
      assert feat_fuse_strategy in ['only_base','concat', 'max', 'avg', 'only_affine']  
      if norm_base:
        base_feat = F.normalize(base_feat) #  compute L2 norm   
      if norm_affine:
        affine_feat = F.normalize(affine_feat)
      if feat_fuse_strategy == 'concat':
        feat = torch.cat((base_feat, affine_feat), 1)
      elif feat_fuse_strategy == 'max': # max for each item
        feat= torch.max(base_feat, affine_feat)
      elif feat_fuse_strategy == 'avg':  #avg for each item
        feat = 0.5 * base_feat + 0.5 * affine_feat  # 0.5*a + 0.5 *b  weight
      elif feat_fuse_strategy == 'only_base':
        feat = base_feat # would not use affine feature
      elif feat_fuse_strategy == 'only_affine':
        feat = affine_feat # would not use affine feature
      # 3. normlize feat
      if norm_BaseAffine:
        feat = F.normalize(feat)
        # or use following
        #N, C, H, W = feat.size()
        #feat = F.normalize(feat, 2, dim=1, keepdim=True)
        #feat = feat.repeat((1,C,1,1)) 

      # now use fused feat to run aligned reid
      global_feat = F.avg_pool2d(feat, feat.size()[2:])
      # shape [N, C]
      global_feat = global_feat.view(global_feat.size(0), -1)
      # shape [N, C, H, 1]
      local_feat = torch.mean(feat, -1, keepdim=True)
      local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
      # shape [N, H, c]
      local_feat = local_feat.squeeze(-1).permute(0, 2, 1)  
      if hasattr(self, 'fc'):
        logits = self.fc(global_feat)
        return global_feat, local_feat, logits
      return global_feat, local_feat
  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)
