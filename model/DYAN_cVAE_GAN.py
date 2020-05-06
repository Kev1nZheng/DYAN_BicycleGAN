import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from model.newResNet import resnet18
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape

        self.fc = nn.Linear(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        d1 = self.down1(torch.cat((x, z), 1))  # d1 size: torch.Size([1, 64, 64, 64])
        d2 = self.down2(d1)  # d2 size: torch.Size([1, 128, 32, 32])
        d3 = self.down3(d2)  # d3 size: torch.Size([1, 256, 16, 16])
        d4 = self.down4(d3)  # d4 size: torch.Size([1, 512, 8, 8])
        d5 = self.down5(d4)  # d5 size: torch.Size([1, 512, 4, 4])
        d6 = self.down6(d5)  # d6 size: torch.Size([1, 512, 2, 2])
        d7 = self.down7(d6)  # d7 size: torch.Size([1, 512, 1, 1])
        u1 = self.up1(d7, d6)  # u1 size: torch.Size([1, 1024, 2, 2])
        u2 = self.up2(u1, d5)  # u2 size: torch.Size([1, 1024, 4, 4])
        u3 = self.up3(u2, d4)  # u3 size: torch.Size([1, 1024, 8, 8])
        u4 = self.up4(u3, d3)  # u4 size: torch.Size([1, 512, 16, 16])
        u5 = self.up5(u4, d2)  # u5 size: torch.Size([1, 256, 32, 32])
        u6 = self.up6(u5, d1)  # u6 size: torch.Size([1, 128, 64, 64])

        return self.final(u6)

##############################
#        Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(
            *list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

##############################
#        Discriminator
##############################
class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)),
            )

        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        x = torch.reshape(x, (1, 10, 128, 128))
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

##############################
#        DYAN
##############################
def creatRealDictionary(T, Drr, Dtheta):
    WVar = []
    Wones = torch.ones(1).cuda()
    Wones = Variable(Wones, requires_grad=False)
    for i in range(0, T):
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))
        W2 = torch.mul(torch.pow(-Drr, i), torch.cos(i * Dtheta))
        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        W4 = torch.mul(torch.pow(-Drr, i), torch.sin(i * Dtheta))
        W = torch.cat((Wones, W1, W2, W3, W4), 0)
        WVar.append(W.view(1, -1))

    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    nG[idx] = np.sqrt(T)
    G = nG
    dic = dic / G
    return dic


def fista(D, Y, lambd, maxIter):
    DtD = torch.matmul(torch.t(D), D)
    L = torch.norm(DtD, 2)
    linv = 1 / L
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(), requires_grad=True)
    t = 1
    y_old = x_old
    lambd = lambd * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(), requires_grad=True) - torch.mul(DtD, linv)

    DtY = torch.mul(DtY, linv)

    Softshrink = nn.Softshrink(lambd)
    with torch.no_grad():
        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old)
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY))
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            tt = (t - 1) / t_new
            y_old = torch.mul(x_new, (1 + tt))
            y_old -= torch.mul(x_old, tt)
            if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new
            del x_new
    return x_old


class DYAN_Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T):
        super(DYAN_Encoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta)
        sparsecode = fista(dic, x, 0.1, 100)
        return Variable(sparsecode)


class DYAN_Decoder(nn.Module):
    def __init__(self, rr, theta, T, PRE):
        super(DYAN_Decoder, self).__init__()
        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE

    def forward(self, x):
        dic = creatRealDictionary(self.T + self.PRE, self.rr, self.theta)
        result = torch.matmul(dic, x)
        return result