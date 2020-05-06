from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, cv2
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from data.kth import KTH_LIST_img_dyan
import skimage

rootDir = '/data/huaiyu/data/kth_action_full_exp/e3d_action_full_64'

dyan_model = torch.load('/data/huaiyu/DYAN/DYAN_kth/weight/e3d_action_full_64/64_Model_3.pth').cuda()
dyan_model.eval()

experiment_index = 'kth_action_full_64_v7_1'
testingData = torch.utils.data.DataLoader(KTH_LIST_img_dyan(rootDir, 10, 'test'), batch_size=1, shuffle=False)
model_id = '195'

Encoder = torch.load('./checkpoint/' + experiment_index + '/Encoder_' + model_id + '.pt').cuda()
Generator = torch.load('./checkpoint/' + experiment_index + '/Generator_' + model_id + '.pt').cuda()
saveDir = './results/' + experiment_index


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(
        torch.cuda.FloatTensor(np.random.normal(0, 1, (mu.size(0), 128))))
    z = sampled_z * std + mu
    return sampled_z


all_sum_psnr = 0
with torch.no_grad():
    for i, sample in enumerate(testingData):

        Encoder.eval()
        Generator.eval()

        real_A = sample['real_A'].cuda()
        real_B = sample['real_B'].cuda()

        folderName = sample['Name'][0].split('/')[-2] + '_%05d' % int(
            sample['Name'][0].split('/')[-1][-9:-5])
        imgPath = os.path.join(saveDir, folderName)

        if not os.path.exists(imgPath):
            os.makedirs(imgPath)

        real_A = torch.reshape(real_A, (1, 64, 128, 128))
        mus, logvar = Encoder(real_A)
        encoded_z = reparameterization(mus, logvar)

        fake_B = Generator(real_A, encoded_z)
        fake_frames_B = dyan_model.forward3(fake_B)

        frames_B = dyan_model.forward3(real_B)

        fake_frames_B = fake_frames_B.cpu().numpy().reshape(10, 128, 128)
        frames_B = frames_B.cpu().numpy().reshape(10, 128, 128)

        sum_psnr = 0

        for j in range(10):
            img_r = frames_B[j, :, :]
            amin_r, amax_r = img_r.min(), img_r.max()
            nor_r = (img_r - amin_r) / (amax_r - amin_r)
            nor_r = nor_r * 255
            cv2.imwrite(imgPath + '/real_B_{:d}.jpg'.format(j), nor_r)

            img_f = fake_frames_B[j, :, :]
            amin_f, amax_f = img_f.min(), img_f.max()
            nor_f = (img_f - amin_f) / (amax_f - amin_f)
            nor_f = nor_f * 255
            cv2.imwrite(imgPath + '/fake_B_{:d}.jpg'.format(j), nor_f)

            psnr = skimage.measure.compare_psnr(nor_r, nor_f, 255)
            sum_psnr += psnr

        sum_psnr = sum_psnr / 10
        all_sum_psnr += sum_psnr
        psnr_log = 'clips:{}, psnr:{}, avg_psnr:{}\n'.format(i, sum_psnr, all_sum_psnr / (i + 1))
        print(psnr_log)
        file = open('./results/log_' + experiment_index + '.txt', mode='a+')
        file.write(psnr_log)
        file.close()
