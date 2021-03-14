from __future__ import absolute_import

import torch
from torch.autograd import Variable

from .metric_learning import get_metric
import numpy as np


class DistanceMetric(object):
    def __init__(self, algorithm='kissme', *args, **kwargs):
        super(DistanceMetric, self).__init__()
        self.algorithm = algorithm
        self.metric = get_metric(algorithm, *args, **kwargs)

    def train(self, feats, labels):
        self.metric.fit(feats, labels)

    def transform(self, X):
        if torch.is_tensor(X):
            X = X.numpy()
            X = self.metric.transform(X)
            X = torch.from_numpy(X)
        else:
            X = self.metric.transform(X)
        return X

