import os

path_name = './bounding_box_train_generate'  # the path of generate images from PN-GAN
new_path_name = './bounding_box_train'# the new path to save those images
if not os.path.exists(new_path_name):
    os.makedirs(new_path_name)
images = os.listdir(path_name)
for k in images:
    if '.jpg' in k or '.png' in k:
        num = 90
        if '_to_cannonical_1' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_1')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_2' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_2')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 1)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 1)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_3' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_3')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 2)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 2)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_4' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_4')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 3)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 3)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_5' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_5')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 4)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 4)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_6' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_6')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 5)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 5)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        elif '_to_cannonical_7' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_7')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 6)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 6)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp, ))
        elif '_to_cannonical_8' in k:
            ids, cam, frame, idx = k.split('_to_cannonical_8')[0].split('_')
            image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 7)
            while True:
                current_images = os.listdir(new_path_name)
                if image_temp not in current_images:
                    break
                num += 8
                image_temp = ids + '_' + cam + '_' + frame + '_{}.jpg'.format(num + 7)
            os.rename(os.path.join(path_name, k), os.path.join(new_path_name, image_temp))
        else:
            pass
            # os.rename(os.path.join(path_name, k), os.path.join(new_path_name, k)) # skip raw image
