from __future__ import print_function
import argparse
import os
import random
from locale import normalize

from matplotlib import pyplot as plt
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split

from torch.autograd import Variable

from BrainDataset import BrainDataset
from model import _netlocalD, _netG
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import utils



parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True ,action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

opt = parser.parse_args()
torch.cuda.empty_cache()

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
train_data = None

device = 'cuda:0'
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.ToTensor()])
    validation_split = .2
    test_split = .2

    dataset_size = 216 * 1000
    sagital_data = BrainDataset("Sagital", dataset_size)
    coronal_data = BrainDataset("Coronal", dataset_size)
    total_data = torch.utils.data.ConcatDataset([sagital_data, coronal_data])

    train_data, _ = random_split(total_data, [800 * 216 * 2, 200 * 216 * 2])
    train_data, _ = random_split(train_data, [600 * 216 * 2, 200 * 216 * 2])

    # dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, prefetch_factor=2)
    # validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=opt.batchSize, shuffle=True,
    #                                                 prefetch_factor=2)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, prefetch_factor=2)

    #
#     dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))
dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, prefetch_factor=2)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)


netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
# input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_masked = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

# real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_masked, label = input_real.cuda(), input_masked.cuda(), label.cuda()
    # real_center = real_center.cuda()


input_real = Variable(input_real)
input_masked = Variable(input_masked)
label = Variable(label)


# real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# MASK = [[1] * 128] * 128
# MASK = np.array(MASK)
# mask = np.zeros(MASK.shape[0])
# for i in range(MASK.shape[0]):
#     if (i % 6) == 0:
#         MASK[i, :] = mask
#         MASK[i + 1, :] = mask
#         MASK[i + 2, :] = mask

MASK = [[1] * 128] * 128
MASK = np.array(MASK)
mask = np.zeros(MASK.shape[1])
for i in range(MASK.shape[0] - 2):
    if (i % 6) == 0:
        MASK[i, :] = mask
        if i + 1 < MASK.shape[0]:
            MASK[i + 1, :] = mask
        if i + 2 < MASK.shape[0]:
            MASK[i + 2, :] = mask

masks = np.array([MASK] * opt.batchSize)
masks = np.float32(masks)
masks = np.reshape(masks, (opt.batchSize, 1, 128, 128))
masks = torch.tensor(masks).to(device)

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)

train_writer = SummaryWriter('logs/edge/train')

for epoch in range(resume_epoch, opt.niter):
    gen_loss_train = []
    dis_loss_train = []
    psnr_train = []
    ssim_train = []
    for i, data in enumerate(dataloader, 0):
        real_cpu, real_masked = data
        t = T.Resize(128)
        real_cpu = t(real_cpu)
        # print(real_cpu.unique())
        real_masked = t(real_masked)
        # real_cpu = real_cpu.to('cuda')

        # real_center_cpu = real_cpu[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
        batch_size = real_cpu.size(0)

        # input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        # input_masked.data.resize_(real_masked.size()).copy_(real_masked)

        input_real.copy_(real_cpu)
        input_masked.copy_(real_masked)
        # print(input_real.data.unique())

        # print(input_real.shape)
        # plt.figure(figsize=(15, 5))
        # plt.style.use('grayscale')
        # plt.subplot(141)
        # plt.imshow(input_real.cpu()[1][0])
        # plt.subplot(142)
        # plt.imshow(input_masked.cpu()[1][0])
        # plt.show()

        # real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        # input_cropped.data[:,0,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*117.0/255.0 - 1.0
        # input_cropped.data[:,1,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*104.0/255.0 - 1.0
        # input_cropped.data[:,2,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*123.0/255.0 - 1.0

        # train with real
        netD.zero_grad()
        label.data.resize_(batch_size).fill_(real_label)

        output = (netD(input_real)).squeeze(1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        inputs = torch.cat((input_masked, masks), dim=1)
        fake = netG(inputs)

        # normalize the tensor values from 0 to 1
        max_val = torch.max(input_real)
        min_val = torch.min(input_real)
        input_real = torch.div(input_real - min_val, max_val - min_val)

        # normalize the tensor values from 0 to 1
        max_val = torch.max(fake)
        min_val = torch.min(fake)
        fake = torch.div(fake - min_val, max_val - min_val)

        outputs_merged = (fake * masks) + (input_real * (1 - masks))

        label.data.fill_(fake_label)
        output = netD(outputs_merged.detach()).squeeze()
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        dis_loss_train.append(errD.item())
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(outputs_merged).squeeze()
        errG_D = criterion(output, label)
        # errG_D.backward(retain_variables=True)

        # errG_l2 = criterionMSE(fake,real_center)
        wtl2Matrix = input_real.clone()
        wtl2Matrix.data.fill_(wtl2)
        # wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred)] = wtl2
        
        errG_l2 = (outputs_merged-input_real).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2
        gen_loss_train.append(errG.item())

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()
        psnr_in = psnr(outputs_merged, input_real)
        ssim_in = ssim(outputs_merged, input_real)

        psnr_train.append(psnr_in.item())
        ssim_train.append(ssim_in.item())

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data.item(), errG_D.item(),errG_l2.item(), D_x,D_G_z1, ))
        if i % 100 == 0:
            vutils.save_image(input_real.data,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch), normalize=True)
            vutils.save_image(input_masked.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch), normalize=True)
            # recon_image = output.clone()
            # recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = fake.data
            vutils.save_image(outputs_merged.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch), normalize=True)
    train_writer.add_scalar('Generator_Loss/Train/Epoch', gen_loss_train, epoch + 1)
    train_writer.add_scalar('Discriminator_Loss/Train/Epoch', dis_loss_train, epoch + 1)

    train_writer.add_scalar('PSNR/Train/Epoch', psnr_train, epoch + 1)
    train_writer.add_scalar('SSIM/Train/Epoch', ssim_train, epoch + 1)

    # do checkpointing
    torch.save({'epoch':epoch+1,
                'state_dict':netG.state_dict()},
                'model/netG_streetview.pth' )
    torch.save({'epoch':epoch+1,
                'state_dict':netD.state_dict()},
                'model/netlocalD.pth' )
