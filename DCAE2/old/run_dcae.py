import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from DCAE import Encoder, Decoder
import torch.nn as nn

#---------------
#Global Variables
#---------------

image_size = 512
z_dimension = 100
image_channels = 3
batch_size = 16
workers = 4

#---------------
#Functions
#---------------


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#---------------
#Nets
#---------------

encoder = Encoder(nc=image_channels, ndf=16, nout=z_dimension)
decoder = Decoder(nz=z_dimension, ngf=18, nc=image_channels)

nets = [encoder, decoder]

for net in nets:
    net.build()
    net.cuda()
    net.apply(init_weights)


#---------------
#Data/Checkpointing
#---------------

dataroot = '/home/tyler/Datasets/Plans_All'
base_dir = '/home/tyler/git/pytorch-experiments/DCAE2/'
checkpoint_path = base_dir + 'checkpoints/'
save_dir = base_dir + 'images/'

dirs = [base_dir, checkpoint_path, save_dir]
for D in dirs:
    if not os.path.exists(D):
        os.mkdir(D)

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Scale(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))

#---------------
#Loss/Optimizers
#---------------

criterion = nn.BCELoss(size_average=True).cuda()
