# -*- coding:utf-8 -*-

from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.  擦除部分(矩形)的最小高宽比
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        # probability的概率会执行擦除操作，否则返回原图像
        # 非擦除操作
        if random.uniform(0, 1) > self.probability:
            return img

        # 擦除操作
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]  # 高*宽=面积 
       
            target_area = random.uniform(self.sl, self.sh) * area  # 随机擦除部分的面积(指定范围内随机挑选的一个面积)
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # 确定一个擦除部分的高宽比

            h = int(round(math.sqrt(target_area * aspect_ratio))) # 得到擦除矩形的高
            w = int(round(math.sqrt(target_area / aspect_ratio))) # 宽

            if w < img.shape[1] and h < img.shape[0]:  # 宽<img宽 且 高<img高则执行后续擦除操作
                # 随机选择出一个pixel点作为擦除矩形的左上角坐标
                x1 = random.randint(0, img.shape[0] - h) 
                y1 = random.randint(0, img.shape[1] - w)
               
                img.flags.writeable = True  # 防止传入的变量是只读模式,导致后面的部分不能修改
                if img.shape[2] == 3:  # 彩图
                    # 分channel擦除一个矩形(这个区域的灰度值均设置为预设的平均值)====>一种数据增强方法
                    # 不同channel的均值不一样是为了克服算法对不同颜色的敏感度不同
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                else:  # 灰度图
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                img.flags.writeable = False
                return img

        return img
