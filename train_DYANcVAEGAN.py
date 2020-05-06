from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model.utils import gridRing
from model.DYAN_cVAE_GAN import DYAN_Encoder, DYAN_Decoder, Encoder, Generator, MultiDiscriminator
from data.kth import KTH_e3dLIST_dyan, KTH_LIST_img_dyan
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data
import torch
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--epochs', type=int, default=300, metavar='N')
parser.add_argument('--seed', type=int, default=1, metavar='S')
args = parser.parse_args()
torch.manual_seed(args.seed)

##############################
#        H_Parameter
##############################
experiment_index = 'kth_action_full'
##############################
#        Data
##############################
rootDir = '/data/huaiyu/data/kth_action_full_exp/e3d_action_full'
trainingData = torch.utils.data.DataLoader(KTH_LIST_img_dyan(rootDir, 10, 'train'), batch_size=1, shuffle=True)
##############################
#        model
##############################

Encoder = Encoder(latent_dim=128).cuda()
Generator = Generator(latent_dim=128, img_shape=[64, 128, 128]).cuda()
D_VAE = MultiDiscriminator(input_shape=[10, 128, 128]).cuda()
D_LR = MultiDiscriminator(input_shape=[10, 128, 128]).cuda()
dyan_model = torch.load('/data/huaiyu/DYAN/DYAN_kth/weight/e3d_action_full_64/64_Model_3.pth').cuda()
dyan_model.eval()

optimizer_E = optim.Adam(Encoder.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_G = optim.Adam(Generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D_VAE = optim.Adam(D_VAE.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D_LR = optim.Adam(D_LR.parameters(), lr=1e-4, betas=(0.5, 0.999))

mae_loss = torch.nn.L1Loss(reduction='sum')
mse_loss = torch.nn.MSELoss(reduction='sum')

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (mu.size(0), 128))))
    z = sampled_z * std + mu
    return z

def get_random_z(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (mu.size(0), 128))))
    return sampled_z
##############################
#        train
##############################
def train(epoch):
    Encoder.train()
    Generator.train()
    valid = 1
    fake = 0
    sum_loss_D_VAE = 0
    sum_loss_D_LR = 0
    sum_loss_GE = 0
    sum_loss_pixel = 0
    sum_loss_kl = 0
    sum_loss_z_L1 = 0
    for batch_idx, sample in enumerate(trainingData):
        real_A = sample['real_A'].cuda()
        real_B = sample['real_B'].cuda()
        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------
        mus, logvar = Encoder(real_B)

        encoded_z = reparameterization(mus, logvar)
        random_z = get_random_z(mus, logvar)

        fake_B = Generator(real_A, encoded_z)
        fake_B_random = Generator(real_A, random_z)

        fake_frames_B = dyan_model.forward3(fake_B)
        fake_frames_B_random = dyan_model.forward3(fake_B_random)
        frames_B = dyan_model.forward3(real_B)

        mus2, logvar2 = Encoder(fake_B_random)

        # update G and E
        D_VAE.set_requires_grad = False
        D_LR.set_requires_grad = False

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        # 1, G(A) should fool D (# Adversarial loss)
        loss_VAE_GAN = D_VAE.compute_loss(fake_frames_B, valid)
        loss_LR_GAN = D_LR.compute_loss(fake_frames_B_random, valid)
        # 2. KL loss (# Kullback-Leibler divergence of encoded B)
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mus ** 2 - logvar - 1)
        # 3, reconstruction |fake_B-real_B| (# Pixelwise loss of translated image by VAE)
        loss_pixel = mse_loss(fake_frames_B, frames_B)

        loss_GE = loss_VAE_GAN + loss_LR_GAN + 0.1 * loss_kl + 10.0 * loss_pixel
        loss_GE.backward(retain_graph=True)

        optimizer_E.step()
        optimizer_G.step()

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        # update G only
        loss_z_L1 = mae_loss(mus2, random_z) * 0.5
        loss_z_L1.backward()
        optimizer_G.step()

        D_VAE.set_requires_grad = True
        D_LR.set_requires_grad = True

        optimizer_D_VAE.zero_grad()
        loss_D_VAE = D_VAE.compute_loss(frames_B, valid) + D_VAE.compute_loss(fake_frames_B.detach(), fake)
        loss_D_VAE.backward(retain_graph=True)
        optimizer_D_VAE.step()

        optimizer_D_LR.zero_grad()
        loss_D_LR = D_LR.compute_loss(frames_B, valid) + D_LR.compute_loss(fake_frames_B_random.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

        if batch_idx % 100 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent:%f ]"
                % (epoch, args.epochs, batch_idx, len(trainingData),
                   loss_D_VAE.item(), loss_D_LR.item(), loss_GE.item(), loss_pixel.item(), loss_kl.item(),
                   loss_z_L1.item()))
        sum_loss_D_VAE += loss_D_VAE.item()
        sum_loss_D_LR += loss_D_LR.item()
        sum_loss_GE += loss_GE.item()
        sum_loss_pixel += loss_pixel.item()
        sum_loss_kl += loss_kl.item()
        sum_loss_z_L1 += loss_z_L1.item()

    loss_log = "\r[Epoch %d/%d ]  [D VAE_loss: %f LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent:%f ]" % (
        epoch, args.epochs,
        sum_loss_D_VAE / len(trainingData),
        sum_loss_D_LR / len(trainingData),
        sum_loss_GE / len(trainingData),
        sum_loss_pixel / len(trainingData),
        sum_loss_kl / len(trainingData),
        sum_loss_z_L1 / len(trainingData))
    print(loss_log)
    file = open('./log/log_' + experiment_index + '.txt', mode='a+')
    file.write(loss_log)
    file.close()
    if not os.path.exists('./checkpoint/' + experiment_index):
        os.makedirs('./checkpoint/' + experiment_index)

    if epoch % 5 == 0:
        torch.save(Encoder, './checkpoint/' + experiment_index + '/Encoder_{:0>3d}.pt'.format(epoch))
        torch.save(Generator, './checkpoint/' + experiment_index + '/Generator_{:0>3d}.pt'.format(epoch))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        if not os.path.exists('./results/' + experiment_index):
            os.makedirs('./results/' + experiment_index)
        if not os.path.exists('./checkpoint/' + experiment_index):
            os.makedirs('./checkpoint/' + experiment_index)
        train(epoch)
