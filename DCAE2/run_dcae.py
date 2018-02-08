import torch
from torch.autograd import Variable
from DCAE import Encoder, Decoder
# import torch.nn as nn

batch_size = 1

encoder = Encoder(nc=3, ndf=8, nout=100)
encoder.build()
decoder = Decoder(nz=100, ngf=8, nc=3)
decoder.build()

encoder.cuda()
decoder.cuda()

x = Variable(torch.randn(batch_size, 3, 512, 512)).cuda()

out = encoder(x)
out = out.view(batch_size, -1, 1, 1)
out = decoder(out, mode='ae')
