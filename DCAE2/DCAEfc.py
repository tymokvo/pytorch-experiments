import torch.nn as nn
import copy
import math
import torchvision.utils as vutils
import torchvision.transforms as transforms
import random
from PIL import Image

imsize = 512
to_image = transforms.ToPILImage()
to_tensor = transforms.Compose([transforms.Resize(imsize, interpolation=Image.NEAREST), transforms.ToTensor()])


class DCAE(nn.Module):
    def __init__(self, nf, z_d=1000):
        super(DCAE, self).__init__()
        #goal: (1 x 3 x 512 x 512) -> (1 x 1000)
        self.z_dim = z_d
        self.encoder = nn.Sequential(
            nn.Conv2d(3, nf * 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            #256 x 256
            nn.Conv2d(nf * 64, nf * 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(nf * 32),
            #128 x 128
            nn.Conv2d(nf * 32, nf * 16, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            #64 x 64
            nn.Conv2d(nf * 16, nf * 8, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(nf * 8),
            #32 x 32
            nn.Conv2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            #16 x 16
            nn.Conv2d(nf * 8, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(nf * 4),
            #8 x 8
            nn.Conv2d(nf * 4, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            #4 x 4
            nn.Conv2d(nf * 4, z_d, 4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        self.fc = nn.Linear(z_d, z_d)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.decoder = nn.Sequential(
            #batch x z_d
            nn.ConvTranspose2d(z_d, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(inplace=True),
            #2 x 2
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(inplace=True),
            #4 x 4
            nn.ConvTranspose2d(nf * 4, nf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(inplace=True),
            #8 x 8
            nn.ConvTranspose2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(inplace=True),
            #16 x 16
            nn.ConvTranspose2d(nf * 8, nf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(inplace=True),
            #32 x 32
            nn.ConvTranspose2d(nf * 16, nf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(inplace=True),
            #64 x 64
            nn.ConvTranspose2d(nf * 16, nf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 32),
            nn.LeakyReLU(inplace=True),
            #128 x 128
            nn.ConvTranspose2d(nf * 32, nf * 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 64),
            nn.LeakyReLU(inplace=True),
            #256 x 256
            nn.ConvTranspose2d(nf * 64, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x, decode=True):
        x = self.encoder(x)
        if not decode:
            return x.squeeze()
        x = self.linear(x)
        x = self.decoder(x)
        return x

    def linear(self, x):
        x = x.view(-1, self.z_dim)
        x = self.fc(x)
        x = self.lrelu(x)
        x = x.view(x.size()[0], self.z_dim, 1, 1)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0]
        c = m.in_channels
        d = m.out_channels
        n = (k ** 2) * c

        m.weight.data = torch.FloatTensor(d, c, k, k).normal_(mean=0, std=math.sqrt(2 / n))

        if m.bias:
            m.bias.data.zero_()


def list_batch_shuffle(in_list, batch_size, shuffle=True):
    dset_size = len(in_list)
    a = list(range(0, dset_size))
    if shuffle:
        random.shuffle(a)
    b = [a[i:i + batch_size] for i in range(0, len(a), batch_size)]
    return b


def gen_batch_tensor(batch, dataset, image_path, label_path):
    inputs = []
    labels = []
    for i in batch:
        inputs += [to_tensor(Image.open(image_path + dataset[i][0]))]
        labels += [to_tensor(Image.open(label_path + dataset[i][1]))]
    return torch.stack(inputs), torch.stack(labels)


def train_dcae(net, batches, dataset, epochs, criterion, optimizer, print_frequency=10, save_dir=None, segmentation=False):
    best_loss = 1e5
    best_net = None
    #TODO: add LR scheduling
    for epoch in range(epochs):
        for batch_num, batch in enumerate(batches):
            optimizer.zero_grad()

            image, label = gen_batch_tensor(batch, dataset, photos_dir, labels_dir)

            image = Variable(image).cuda()

            output = net(image)

            if segmentation:
                label = Variable(label).cuda()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(output, image)
                loss.backward()
                optimizer.step()

            if loss.data[0] < best_loss:
                best_loss = loss.data[0]
                best_net = copy.deepcopy(net)
                print('!!! new best loss: {:.4f}'.format(best_loss))

            status = 'epoch: [{:4d}/{:4d}] batch: [{:4d}/{:4d}] loss: {:.4f}'
            if print_frequency is not None and batch_num % print_frequency == 0:
                status = status.format(epoch + 1, epochs, batch_num + 1, batches.__len__(), loss.data[0])
                print(status)

        if save_dir is not None:
            grid = vutils.make_grid(output.data.cpu(), int(math.sqrt(image.size(0))), 8, normalize=True)
            grid = to_image(grid)
            grid.save(save_dir + 'e{}_output.jpg'.format(epoch))

            real_grid = vutils.make_grid(image.data.cpu(), int(math.sqrt(image.size(0))), 8, normalize=True)
            real_grid = to_image(real_grid)
            real_grid.save(save_dir + 'b{}_real.jpg'.format(batch_num))
    return best_net


if __name__ == '__main__':
    import os
    import time
    import torch
    import torch.optim as optim
    # import torchvision.datasets as dset
    from torch.autograd import Variable

    nf = 8
    z_d = 100
    batch_size = 4
    workers = 4
    image_size = 512
    n_epochs = 100
    lr = .01

    base_dir = '/home/tyler/git/pytorch-experiments/DCAE2/output/'
    model_dir = base_dir + 'models/'
    save_dir = base_dir + 'images/'


    dirs = [base_dir, model_dir, save_dir]
    for D in dirs:
        assert os.path.exists(D), 'missing a directory! {}'.format(D)


    dataroot = '/home/tyler/Datasets/Facades_Extended/'

    photos_dir = dataroot + 'photos/'
    labels_dir = dataroot + 'labels/'

    photo_list = sorted(os.listdir(photos_dir))
    label_list = sorted(os.listdir(labels_dir))

    photo_label_list = list(zip(photo_list, label_list))

    batches = list_batch_shuffle(photo_label_list, batch_size)

    net = DCAE(nf, z_d=z_d)
    net.apply(init_weights)
    net.float().cuda()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss(size_average=True).cuda()

    net = train_dcae(net, batches, photo_label_list, epochs=100, criterion=criterion, optimizer=optimizer, save_dir=save_dir, segmentation=True)

    day = time.strftime('%Y-%m-%d')
    now = time.strftime('%H-%M-%S')

    netname = '{}_{}_{}.pth'.format(net.__class__.__name__, day, now)

    torch.save(net, model_dir + netname)
