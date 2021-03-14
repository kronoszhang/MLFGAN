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
import sys
import logger


# Save model
def save_model(epoch, path, nets, optimizers, net_name):
    netG, netD = nets
    print(net_name)
    optimizer_G, optimizer_D = optimizers
    print ("Saving model ------------------------------->")
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': netG.state_dict(), 'optimizer' : optimizer_G.state_dict(), },
                f='%s/%s/%s_%d.pkl' % (path, net_name, 'G', epoch))
    torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer' : optimizer_D.state_dict(), },
                f='%s/%s/%s_%d.pkl' % (path, net_name, 'D', epoch))
    print ("Finished ----------------------------------->")


# Save images
def save_images(net_name, epoch, PATH, src_img, pose, tgt_img, fake_img, summary):
    n, c, h, w = src_img.size()
    samples = torch.FloatTensor(4*n, c, h, w).zero_()
    for i in range(n):
        samples[4*i+0] = src_img[i].data
        samples[4*i+1] = pose[i].data
        samples[4*i+2] = tgt_img[i]
        samples[4*i+3] = fake_img[i].data

    images = utils.make_grid(samples, nrow=8, padding=30, normalize=True)
    summary.add_image('samples', images, epoch)
    file_name = os.path.join(PATH, net_name)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    utils.save_image(samples, '%s/samples_%d.png' % (file_name, epoch), nrow=8, padding=30, normalize=True)


# Load Data
def load_data():
    train_data = dataset.Market_DataLoader(imgs_path=cfg.TRAIN.imgs_path, pose_path=cfg.TRAIN.pose_path, idx_path=cfg.TRAIN.idx_path,
                                           transform=dataset.train_transform(), loader=dataset.val_loader) # train_loader ?
    train_loader = Data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                   num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)

    val_data = dataset.Market_DataLoader(imgs_path=cfg.TRAIN.imgs_path, pose_path=cfg.TRAIN.pose_path, idx_path=cfg.TEST.idx_path,
                                         transform=dataset.val_transform(), loader=dataset.val_loader)
    val_loader = Data.DataLoader(val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.TRAIN.NUM_WORKERS)

    train = [train_data, train_loader]
    val = [val_data, val_loader]
    return train, val


# Load Network
def load_network():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    print ('###################################')
    print ("#####      Build Network      #####")
    print ('###################################')

    nets = []
    netG = network.Res_Generator(cfg.TRAIN.ngf, cfg.TRAIN.num_resblock)

    netD = network.Patch_Discriminator(cfg.TRAIN.ndf)

    nets.append(netG)
    nets.append(netD)

    print_networks(nets, debug=cfg.DEBUG)

    for net in nets:
        net.cuda()

    return nets

def print_networks(model_names, debug):
        print ('---------------- Network initialized ----------------')
        names = ['netG', 'netD']
        for i, net in enumerate(model_names):
            num_params = 0
            for param in net.parameters():  # for each layer's `param` in `net`
                num_params += param.numel() # `numel` return num of items in array. see in : https://blog.csdn.net/ccbrid/article/details/86759679
            if debug:
                print ('=========== %s ===========' % names[i])
                print (net)
            print ('[Network %s] Total number of parameters: %.3f M' % (names[i], num_params / 1e6))
        print ('-----------------------------------------------------')


# define optimizers
def Optimizer(nets):
    netG, netD = nets

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=cfg.TRAIN.LR, betas=(0.5, 0.999))# `betas`: coefficients used for computing running averages
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=cfg.TRAIN.LR, betas=(0.5, 0.999))
    optimizers = [optimizer_G, optimizer_D]

    lr_policy = lambda epoch: (1 - 1 * max(0, epoch-cfg.TRAIN.LR_DECAY) / cfg.TRAIN.LR_DECAY)
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_policy) # see `lr_scheduler.LambdaLR` in `torch offical website`
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_policy)
    schedulers = [scheduler_G, scheduler_D]

    summary = SummaryWriter(log_dir='%s/%s' % (os.path.join(cfg.FILE_PATH, 'log'), 
                            cfg.NET), comment='')# see in : https://blog.csdn.net/ShuqiaoS/article/details/88559977

    return optimizers, schedulers, summary


def loss_func():
    criterionGAN = torch.nn.MSELoss().cuda()  # `L_GAN` in Eq.(2) , paper page 4
    criterionIdt = torch.nn.L1Loss().cuda()   # `L_L1` in Eq.(2) , paper page 4
    criterion = [criterionGAN, criterionIdt]

    return criterion


def train(train_file, val_file, nets, optimizers, schedulers, summary, criterion):
    print ('\n###################################')
    print ("#####      Start Traning      #####")
    print ('###################################')

    train_data, train_loader = train_file
    val_data, val_loader = val_file
    netG, netD = nets
    optimizer_G, optimizer_D = optimizers
    scheduler_G, scheduler_D = schedulers
    criterionGAN, criterionIdt = criterion
    
    # resume
    if os.path.exists('./model/GAN/G_2.pkl'):  # if train from epoch 1, this would be useless; if already trained 2 or more epoch, and train again, this would be used to retrain with the base of epoch 2 result
        print('===========REUSING EARLIER RESULT============') 
        checkpoint = torch.load('./model/GAN/G_2.pkl')
        netG.load_state_dict(checkpoint['state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        checkpoint = torch.load('./model/GAN/D_2.pkl')
        netD.load_state_dict(checkpoint['state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        netG.train()
        netD.train()

    count = 0
    for epoch in range(1, cfg.TRAIN.MAX_EPOCH+1):
        scheduler_G.step()  # run one step, see more in `torch offical website`
        scheduler_D.step()
        """
        `step` : iter step, produce by `enumerate`
        if one person have 3 image named '1,2,3', so `self.data[this person] = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]`
        if we choose one pair, like (2,3)
        `src_img` : image1, i.e: 2
        `tgt_img` : image2, i.e: 3
        in real, would used in all images of this person even more many people's, i.e:
        `src_img = [1,1,2,2,3,3]` while `tgt_img = [2,3,1,3,1,2]` accordingly
        """
        for step, (src_img, tgt_img, pose) in enumerate(train_loader):
            begin = time.time()

            # #######################################################
            # (1) Data process
            # #######################################################

            src_img = Variable(src_img).cuda()      # N x 3 x H x W
            tgt_img = Variable(tgt_img).cuda()      # N x 3 x H x W
            pose = Variable(pose).cuda()            # N x 3 x H x W


            # #######################################################
            # (2) Generate images
            # #######################################################
            fake_img = netG(src_img, pose) # N src images and N accordingly pose images to generate N fake image with same persons in src images and same poses in pose images


            # #######################################################
            # (3) Update Generators
            # #######################################################
            D_fake_img = netD(fake_img) # to judge generated N fake images
            G_loss = criterionGAN(D_fake_img, torch.ones_like(D_fake_img))  # MSE loss
            idt_loss = criterionIdt(fake_img, tgt_img) * cfg.TRAIN.lambda_idt  # L1 loss

            loss_G = G_loss + idt_loss   # `L_Gp` in Eq.(2), paper page4
            optimizer_G.zero_grad() # Generator(GAN) bp
            loss_G.backward()
            optimizer_G.step()  # Performs a single optimization step. see in `torch offical website`


            # #######################################################
            # (4) Update Discriminators
            # #######################################################
            D_fake_img = netD(fake_img.detach()) # N fake images to judge
            D_real_img = netD(src_img)  # N real images to judge

            D_fake_loss = criterionGAN(D_fake_img, torch.zeros_like(D_fake_img)) # MSE Loss
            D_real_loss = criterionGAN(D_real_img, torch.ones_like(D_real_img))  # MSE Loss of `D_real_loss` is really small, almostly equ to 0

            loss_D = D_fake_loss + D_real_loss
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()


            # #######################################################
            # (5) Update Log and Display loss info.
            # #######################################################
            count += 1
            summary.add_scalar('G_loss', G_loss.data, count) # Loss_GAN
            summary.add_scalar('Idt_loss', idt_loss.data, count)# Loss_L1
            summary.add_scalar('netG_loss', loss_G.data, count) # Loss_Gp
            summary.add_scalar('netD_loss', loss_D.data, count) # Loss_Dp
            if (step+1)%cfg.TRAIN.DISPLAY==0:
            	print ('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  |  G_loss: {:.6f}  |  D_loss: {:.6f}  |  Idt_loss: {:.6f}  |  Time: {:.3f}'
               		.format(epoch, cfg.TRAIN.MAX_EPOCH, step+1, len(train_loader), optimizer_G.param_groups[0]['lr'],
                     	  loss_G.data, loss_D.data, idt_loss.data, time.time()-begin))


        # #######################################################
        # (7) Validation
        # #######################################################
        netG.eval()
        PATH = os.path.join(cfg.FILE_PATH, 'images')
        
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        
        for _, (src_img, tgt_img, pose) in enumerate(val_loader):
            src_img = Variable(src_img, volatile=True).cuda()  # `volatile=True` means not backward pass, save memory
            pose = Variable(pose, volatile=True).cuda()

            fake_img = netG(src_img, pose)

            save_images(cfg.NET, epoch, PATH, src_img, pose, tgt_img, fake_img, summary)
            break    # For better comparison, we only test one specfic batch images
        netG.train()  # change mode to 'train'


        # #######################################################
        # (7) Save models per epoch
        # #######################################################
        MODEL_PATH = os.path.join(cfg.FILE_PATH, 'model')
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        print(MODEL_PATH)
        save_model(epoch, MODEL_PATH, nets, optimizers, cfg.NET)

    summary.close()


def main():
    sys.stdout = logger.Logger('./log_GAN_ep2.txt')
    train_file, val_file = load_data()
    nets = load_network()
    optimizers, schedulers, summary = Optimizer(nets)  # `schedulers` is the `lr_schedulers` and `summary` is `tensorboardX log`
    criterion = loss_func()
    train(train_file, val_file, nets, optimizers, schedulers, summary, criterion)


if __name__ == '__main__':
    main()

