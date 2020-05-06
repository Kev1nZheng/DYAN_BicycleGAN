############################# Import Section #################################
## Generic imports
import os
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

## Imports related to PyTorch
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


############################# Import Section #################################

## Dataloader for PyTorch.
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, folderList, rootDir, N_FRAME):

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
            Image Size = 240,320
            2 channels : U and V
        """
        self.listOfFolders = folderList
        # self.imageArray = imageArray
        # self.index = index
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 128 * 160  # If Kitti dataset, self.numpixels = 128*160

    def __len__(self):
        return len(self.listOfFolders)

    def readData(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        OF = torch.FloatTensor(3, self.nfra, self.numpixels)
        for framenum in range(self.nfra):
            flow = np.load(os.path.join(path, str(framenum) + '.npy'))
            flow = np.transpose(flow, (2, 0, 1))
            OF[:, framenum] = torch.from_numpy(flow.reshape(2, self.numpixels)).type(torch.FloatTensor)

        return OF

    def readImg(self, folderName):
        sample = torch.FloatTensor(3, self.nfra, self.numpixels)
        value = self.index[folderName]
        nFrames = value['last'] - value['first']
        startid = random.randint(0, nFrames - 20)

        for i in range(startid, startid + self.nfra):
            pix = self.imageArray[i].reshape(128, 160, 3)
            pix = pix / 255.
            pix = np.transpose(pix, (2, 0, 1))
            sample[:, i - startid] = torch.from_numpy(pix.reshape(3, self.numpixels)).type(torch.FloatTensor)
        return sample

    def readClip(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        print(path)
        sample = torch.FloatTensor(3, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
        nFrames = len(frames)
        # nFrames = len([each for each in os.listdir(os.path.join(self.rootDir, folderName[0])) if each.endswith(('.jpg', 'png'))])
        print(nFrames)
        startid = random.randint(0, nFrames - 20)
        for framenum in range(self.nfra):
            imgname = os.path.join(path, '%04d' % (framenum + startid + 1) + '.jpg')
            print(imgname)
            img = Image.open(imgname)
            img = img.resize(128, 160, 3)
            pix = np.array(img) / 255.
            pix = np.transpose(pix, (2, 0, 1))
            sample[:, framenum] = torch.from_numpy(pix.reshape(3, self.numpixels)).type(torch.FloatTensor)
        return sample

    # path = os.path.join(self.rootDir,folderName)
    # IM = torch.FloatTensor(3,self.nfra,self.numpixels)
    # frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
    # nFrames = len(frames)
    # frames.sort()
    # startid = random.randint(0,nFrames - 20)

    # for framenum in range(self.nfra):
    # 	imgname = os.path.join(path,'%05d' % (framenum+startid+1) +'.jpg')
    # 	img = Image.open(imgname)
    # 	img = img.resize(__imgsize__)
    # 	pix = np.array(img)/255.
    # 	pix = np.transpose(pix,(2,0,1))
    # 	IM[:,framenum] = torch.from_numpy(pix.reshape(3,self.numpixels)).type(torch.FloatTensor)
    # return IM

    def __getitem__(self, idx):

        folderName = self.listOfFolders[idx]
        Frame = self.readClip(folderName)
        sample = {'frames': Frame}

        return sample

## Design poles

def gridRing(N):
    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1 - epsilon_low)
    rmax = (1 + epsilon_high)
    thetaMin = 0.001
    thetaMax = np.pi / 2 - 0.001
    delta = 0.001
    Npole = int(N / 4)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)
    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P, -P, np.conjugate(P), np.conjugate(-P)), axis=0)

    return P, Pall


## Generate the grid on poles
def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j * py

    return P


# Create Gamma for Fista
def getWeights(Pall, N):
    g2 = pow(abs(Pall), 2)
    g2N = np.power(g2, N)

    GNum = 1 - g2
    GDen = 1 - g2N
    idx = np.where(GNum == 0)[0]

    GNum[idx] = N
    GDen[idx] = pow(N, 2)
    G = np.sqrt(GNum / GDen)
    return np.concatenate((np.array([1]), G))


## Functions for printing time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


## Function to save the checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def getListOfFolders(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.split('/', expand=True)[1]
    data = data.str.rstrip(".avi").values.tolist()

    return data


def getIndex(sources_list):
    index = {}
    folderName = set(sources_list)
    for folder in folderName:
        first = 0
        last = 0
        for count, name in enumerate(sources_list):
            if folder == name:
                if first == 0:
                    if count == 0:
                        first = count - 1
                    else:
                        first = count

                last = count
                index[folder] = {"first": first, "last": last + 1}
            else:
                continue

    if index['city-2011_09_26_drive_0001_sync']['first'] == -1:
        index['city-2011_09_26_drive_0001_sync']['first'] = 0

    return index
