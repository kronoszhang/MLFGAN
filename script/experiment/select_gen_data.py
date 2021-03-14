# -*- coding: utf-8 -*-
import os
import random
import shutil
import math

ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
imgs_path = '../../market1501_gen/bounding_box_train'

images = os.listdir(imgs_path)
for image in images:
    if '.db' in image:
        images.remove(image)
for image in images:
    assert '.db' not in image

images_num = len(images)
for ratio in ratios:
    select_images_num = math.floor(images_num * ratio)
    b_length = 0
    image_list = []

    while b_length <= select_images_num:
        random_number = random.randint(0, images_num - 1)
        if random_number not in image_list:
            image_list.append(images[random_number])
            b_length = len(image_list) + 1

    save_imgs_path = '../../Dataset/pngan_market_raw_test_{}/bounding_box_train'.format(ratio)
    if not os.path.exists(save_imgs_path):
        os.makedirs(save_imgs_path)


    count = 0
    for image in image_list:
        if select_images_num == 0:
            print(ratio, '▇' * (math.floor(20)) + str(100) + '%')  # 100%时进度条占20格
        else:
            print('\r', ratio, '▇' * (math.floor(count / select_images_num * 20)) +
                  ' ' * (20 - math.floor(count / select_images_num * 20)) +
                  str(count / select_images_num * 100) + '%')

        src_path = os.path.join(imgs_path, image)
        des_path = os.path.join(save_imgs_path, image)
        shutil.copy(src_path, des_path)
        count += 1





# import os
# ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# for ratio in ratios:
#     save_imgs_path = '../../Dataset/pngan_market_raw_test_{}/bounding_box_train'.format(ratio)
#     x = os.listdir(save_imgs_path)
#     for xx in x:
#         os.remove(os.path.join(save_imgs_path, str(xx)))
