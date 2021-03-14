"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid) 
reid/evaluation_metrics/ranking.py. Modifications: 
1) Only accepts numpy data input, no torch is involved.
1) Here results of each query can be returned.
2) In the single-gallery-shot evaluation case, the time of repeats is changed 
   from 10 to 100.
"""
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
  mask = np.zeros(num, dtype=np.bool)
  for _, indices in ids_dict.items():
    i = np.random.choice(indices)
    mask[i] = True
  return mask


def cmc(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    topk=100,
    separate_camera_set=False,
    single_gallery_shot=False,
    first_match_break=False,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
  # all code used for find right matched position of rank list
  # and rank list is producted by `distmat`
  
  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape
  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1) # sort `distmat` , so `indices` would be rank list
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])  # get matched images (id matched)
  # Compute CMC for each query
  ret = np.zeros([m, topk])
  is_valid_query = np.zeros(m)
  num_valid_queries = 0
  for i in range(m): # for m query
    # Filter out the samples from same id and same camera from rank list
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |  # id not same  or
             (gallery_cams[indices[i]] != query_cams[i]))  # cam not same
    if separate_camera_set:
      # Filter out samples from same camera from rank list
      valid &= (gallery_cams[indices[i]] != query_cams[i])  # only cam not same
    # `valid` is a condition , used for check valid samples from `matches`
    # eg: matches[i, valid]====> get query i 's all matched images with condition `valid`
    if not np.any(matches[i, valid]): continue  # 
    is_valid_query[i] = 1 # the query has matched images
    if single_gallery_shot:
      repeat = 100
      gids = gallery_ids[indices[i][valid]]  
      inds = np.where(valid)[0] # valid index
      ids_dict = defaultdict(list)
      for j, x in zip(inds, gids):
        ids_dict[x].append(j)
    else:
      repeat = 1
    for _ in range(repeat):
      if single_gallery_shot:
        # Randomly choose one instance for each id , here `sampled` is a new condition build on `valid`
        # for contain `valid` and `sample: random choose one` 2 conditions 
        sampled = (valid & _unique_sample(ids_dict, len(valid)))  
        index = np.nonzero(matches[i, sampled])[0]  # get matched image for query, only match one for one query
      else:
        index = np.nonzero(matches[i, valid])[0] # get matched images for query
      delta = 1. / (len(index) * repeat)
      for j, k in enumerate(index):
        if k - j >= topk: break
        if first_match_break:
          ret[i, k - j] += 1 # only right matched position add 1
          break
        ret[i, k - j] += delta # # only right matched position add delta
    num_valid_queries += 1
  if num_valid_queries == 0:
    raise RuntimeError("No valid query")
  ret = ret.cumsum(axis=1)
  if average:
    return np.sum(ret, axis=0) / num_valid_queries
  return ret, is_valid_query


def mean_ap(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """


  # all code used for find right matched position of rank list
  # and rank list is producted by `distmat`
  
  
  # -------------------------------------------------------------------------
  # The behavior of method `sklearn.average_precision` has changed since version
  # 0.19.
  # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
  # (https://github.com/zhunzhong07/person-re-ranking/
  # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
  # (http://www.liangzheng.org/Project/project_reid.html).
  # My current awkward solution is sticking to this older version.
  import sklearn
  cur_version = sklearn.__version__
  required_version = '0.18.1'
  if cur_version != required_version:
    print('User Warning: Version {} is required for package scikit-learn, '
          'your current version is {}. '
          'As a result, the mAP score may not be totally correct. '
          'You can try `pip uninstall scikit-learn` '
          'and then `pip install scikit-learn=={}`'.format(
      required_version, cur_version, required_version))
  # -------------------------------------------------------------------------

  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape

  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1)  # rank list
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])  # matched images
  # Compute AP for each query
  aps = np.zeros(m)
  is_valid_query = np.zeros(m)
  for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    y_true = matches[i, valid]  # matched images with condition `valid`  ==>i.e. cross cam
    y_score = -distmat[i][indices[i]][valid]  # rank list score with condition `valid`  
    if not np.any(y_true): continue
    is_valid_query[i] = 1
    aps[i] = average_precision_score(y_true, y_score)
  if len(aps) == 0:
    raise RuntimeError("No valid query")
  if average:
    return float(np.sum(aps)) / np.sum(is_valid_query)
  return aps, is_valid_query