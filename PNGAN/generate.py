import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from config import cfg
from tensorboardX import SummaryWriter
import os, itertools
import network
import dataset
import time
import matplotlib.pyplot as plt


# Save images
def save_images(name, PATH, fake_img, src_img):
    samples = fake_img.data
    raws = src_img.data
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    utils.save_image(samples, '%s/%s.png' % (PATH, name), nrow=1, padding=0, normalize=True)
    # save raw image
    utils.save_image(raws, '%s/%s.png' % (PATH, name.split('_to_')[0]), nrow=1, padding=0, normalize=True)
    

# Load Data
def load_data():
    datasets = []
    for sub_dir in os.listdir(cfg.GENERATE.imgs_path):
        if '.txt' not in sub_dir: # skip `readme.txt` file in Market dataset
            val_data = dataset.Market_generate(imgs_path=os.path.join(cfg.GENERATE.imgs_path,sub_dir), 
                                        pose_path=cfg.GENERATE.pose_path,
                                        transform=dataset.val_transform(), 
                                        loader=dataset.val_loader)
            val_loader = Data.DataLoader(val_data, batch_size=cfg.GENERATE.BATCH_SIZE, shuffle=False,
                                        num_workers=cfg.GENERATE.BATCH_SIZE)

            val = [val_data, val_loader, sub_dir]
            datasets.append(val)
    return datasets


# Load Network
def load_network(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID
    print ('###################################')
    print ("#####      Load Network      #####")
    print ('###################################')

    nets = []
    netG = network.Res_Generator(cfg.TRAIN.ngf, cfg.TRAIN.num_resblock)
    netG.load_state_dict(torch.load(model_path)['state_dict'])

    nets.append(netG)
    for net in nets:
        net.cuda()

    print ('Finished !')
    return nets



def test(datasets, nets):
    print ('\n###################################')
    print ("#####      Start Testing      #####")
    print ('###################################')
    
    assert len(datasets) > 0
    for data in datasets:  # for every folder in Market1501
        _, val_loader, sub_dir = data
        save_path = sub_dir + '_generate'
        netG = nets[0]

        for _, (src_img, pose, name) in enumerate(val_loader):
            # #######################################################
            # (1) Data process
            # #######################################################
            src_img = Variable(src_img, volatile=True).cuda()      # N x 3 x H x W
            pose = Variable(pose, volatile=True).cuda()            # N x 3 x H x W
 

            # #######################################################
            # (2) Generate images
            # #######################################################
            fake_img = netG(src_img, pose)


            # #######################################################
            # (3) Save images
            # #######################################################
            save_images(name[0], save_path, fake_img, src_img)
            print ('Generate image: ', name[0])



def main(model_path):
    datasets = load_data()
    nets = load_network(model_path)
    test(datasets, nets)


if __name__ == '__main__':
    model_path =  'model/GAN/G_20.pkl'      # 'model/YOUR_SAVED_MODEL'
    main(model_path)

