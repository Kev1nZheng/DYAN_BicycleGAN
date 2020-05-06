import torch
import torch.utils.data as data
import numpy as np
import os
import random
import cv2

img_channel = 2


class moving_mnist_data(data.Dataset):
    def __init__(self, folderList, rootDir, N_FRAME):
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 64 * 64

        self.dir_A = '/data/Huaiyu/DYAN/Moving_MNIST/MovingSymbols2_same_4px_OF_DYAN/train/A'
        self.dir_B = '/data/Huaiyu/DYAN/Moving_MNIST/MovingSymbols2_same_4px_OF_DYAN/train/B'
        self.A_paths = os.listdir(self.dir_A)
        self.B_paths = os.listdir(self.dir_B)
        self.A_paths.sort()
        self.B_paths.sort()

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print('self.A_size', self.A_size)

    def __len__(self):
        return self.A_size

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_path = os.path.join(self.dir_A, A_path)
        A_img = np.load(A_path)
        # print('A_img', A_img.shape)
        A_img_X = A_img[0, :, :]
        # A_img_Y = A_img[1, :, :]
        # A_img = np.concatenate((A_img_X, A_img_Y))
        A_img = np.resize(A_img, (161, 64, 64))
        A_img.astype(float)
        A_img = torch.from_numpy(A_img).cuda()

        # print('A_img', A_img.shape)
        return A_img


if __name__ == '__main__':
    rootDir = '/data/Huaiyu/DYAN/Moving_MNIST/Frames'
    folderList = []

    trainingData = moving_mnist_data(folderList, rootDir, 10)
    for i, input in enumerate(trainingData):
        print('input frames_A', input.shape)
        print(i)
