import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize = opt.latenSize
        self.disc_size = opt.latenSize
        self.use_mi = opt.use_mi
        if opt.use_mi:
            self.disc_size = opt.latenSize*2
        self.encoder_linear = nn.Linear(opt.resSize, self.disc_size)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def reparameter(self, mu, sigma):
        return (torch.randn_like(mu) *sigma) + mu

    def forward(self, x, train_G=False):
        laten=self.lrelu(self.encoder_linear(x))
        if self.use_mi:
            mus,stds = laten[:,:self.latensize],laten[:,self.latensize:]
            stds=self.sigmoid(stds)
            encoder_out = self.reparameter(mus, stds)
        else:
            mus, stds, encoder_out = laten, laten, laten
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)

        return mus,stds,dis_out



