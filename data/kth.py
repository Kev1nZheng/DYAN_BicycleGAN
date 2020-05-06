import torch
import torch.utils.data as data
import numpy as np
import os
import random
import cv2

img_channel = 1


class KTH(data.Dataset):
    def __init__(self, rootDir):
        self.dir_A = rootDir
        self.dir_B = rootDir.replace('A/test', 'B/test')
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
        A_path = self.A_paths[
            index % self.A_size]  # make sure index is within then range
        A_path = os.path.join(self.dir_A, A_path)
        B_path = A_path.replace('A/test', 'B/test')
        A_img = np.load(A_path)
        B_img = np.load(B_path)
        return A_img, B_img


class KTH_img(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 64 * 64
        classesList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
        classesList.sort()
        self.listOfFolders = []
        for i in range(len(classesList)):
            classesList[i] = os.path.join(rootDir, classesList[i])
            classes_videoList = [name for name in os.listdir(classesList[i]) if
                                 os.path.isdir(os.path.join(classesList[i]))]
            classes_videoList.sort()
            for j in range(len(classes_videoList)):
                classes_videoList[j] = os.path.join(classesList[i], classes_videoList[j])
            self.listOfFolders.extend(classes_videoList)
        self.listOfFolders.sort()

    def __len__(self):
        return len(self.listOfFolders)

    def readClip(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.png')]
        nFrames = len(frames)
        startid = random.randint(0, nFrames - 40)
        for framenum in range(self.nfra):
            imgname_A = os.path.join(path, 'image-%03d_64x64' % (framenum + startid + 1) + '.png')
            imgname_B = os.path.join(path, 'image-%03d_64x64' % (framenum + startid + 11) + '.png')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
            pix_A = np.array(img_A)
            pix_B = np.array(img_B)
            pix_A = pix_A / 225
            pix_B = pix_B / 225
            pix_A = pix_A[:, :, np.newaxis]
            pix_B = pix_B[:, :, np.newaxis]
            pix_A = np.transpose(pix_A, (2, 0, 1))
            pix_B = np.transpose(pix_B, (2, 0, 1))
            sample_A[:, framenum] = torch.from_numpy(pix_A.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(pix_B.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame_A, Frame_B = self.readClip(folderName)
        sample = {'frames_A': Frame_A, 'frames_B': Frame_B, 'folderName': folderName}
        return sample


class KTH_img_dyan(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.imgDir = '/data/Huaiyu/DYAN/data/kth/processed'
        self.nfra = N_FRAME
        self.numpixels = 64 * 64
        classesList = [name for name in os.listdir(self.imgDir) if os.path.isdir(os.path.join(self.imgDir))]
        classesList.sort()
        self.listOfFolders = []
        for i in range(len(classesList)):
            classesList[i] = os.path.join(self.imgDir, classesList[i])
            classes_videoList = [name for name in os.listdir(classesList[i]) if
                                 os.path.isdir(os.path.join(classesList[i]))]
            classes_videoList.sort()
            for j in range(len(classes_videoList)):
                classes_videoList[j] = os.path.join(classesList[i], classes_videoList[j])
            self.listOfFolders.extend(classes_videoList)
        self.listOfFolders.sort()

    def __len__(self):
        return len(self.listOfFolders)

    def readClip(self, folderName):
        path = os.path.join(self.imgDir, folderName)
        real_A_Dir = os.path.join(self.rootDir, 'A/train/')
        real_B_Dir = os.path.join(self.rootDir, 'B/train/')
        real_A_path = os.path.join(real_A_Dir, folderName.split('/')[-1] + '.npy')
        real_B_path = os.path.join(real_B_Dir, folderName.split('/')[-1] + '.npy')
        real_A = np.load(real_A_path)
        real_B = np.load(real_B_path)
        real_A = np.reshape(real_A, (161, 64, 64))
        real_B = np.reshape(real_B, (161, 64, 64))
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.png')]
        nFrames = len(frames)
        startid = 10
        for framenum in range(self.nfra):
            imgname_A = os.path.join(path, 'image-%03d_64x64' % (framenum + startid) + '.png')
            imgname_B = os.path.join(path, 'image-%03d_64x64' % (framenum + startid + 10) + '.png')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
            pix_A = np.array(img_A)
            pix_B = np.array(img_B)
            pix_A = pix_A / 225
            pix_B = pix_B / 225
            pix_A = pix_A[:, :, np.newaxis]
            pix_B = pix_B[:, :, np.newaxis]
            pix_A = np.transpose(pix_A, (2, 0, 1))
            pix_B = np.transpose(pix_B, (2, 0, 1))
            sample_A[:, framenum] = torch.from_numpy(pix_A.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(pix_B.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B, real_A, real_B

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame_A, Frame_B, real_A, real_B = self.readClip(folderName)
        sample = {
            'frames_A': Frame_A,
            'frames_B': Frame_B,
            'real_A': real_A,
            'real_B': real_B,
            'Name': folderName
        }
        return sample


class KTH_LIST_dyan(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 64 * 64
        with open('/data/Huaiyu/DYAN/DYAN_cVAE_GAN/data/your_file.txt', 'r') as f:
            self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def readClip(self, folderName, startid):
        path_A = folderName.split('/')[-1] + '_%05d.npy' % startid
        path_A = os.path.join(self.rootDir, 'A/train', path_A)
        path_B = path_A.replace('/A/', '/B/')

        real_A = np.load(path_A).squeeze(0)
        real_B = np.load(path_B).squeeze(0)

        return real_A, real_B

    def __getitem__(self, idx):
        folderName = os.path.dirname(self.list[idx])
        startid = int(self.list[idx].split('/')[-1])
        Frame_A, Frame_B = self.readClip(folderName, startid)
        sample = {
            'frames_A': Frame_A,
            'frames_B': Frame_B,
            'Name': self.list[idx]
        }
        return sample


class KTH_e3dLIST_dyan(data.Dataset):
    def __init__(self, rootDir, N_FRAME, mode):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 128 * 128
        self.mode = mode
        if mode == 'train':
            with open('./data/e3d_kth_train_list_mini.txt', 'r') as f:
                self.list = f.readlines()
        if mode == 'test':
            with open('./data/e3d_kth_test_list_mini.txt', 'r') as f:
                self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def readClip(self, folderName, startid):
        path_A = folderName.split('/')[-1] + '_%05d.npy' % startid
        if self.mode == 'train':
            path_A = os.path.join(self.rootDir, 'A/train', path_A)
        if self.mode == 'test':
            path_A = os.path.join(self.rootDir, 'A/test', path_A)
        path_B = path_A.replace('/A/', '/B/')

        real_A = np.load(path_A).squeeze(0)
        real_B = np.load(path_B).squeeze(0)

        return real_A, real_B

    def __getitem__(self, idx):
        folderName = os.path.dirname(self.list[idx])
        startid = int(self.list[idx].split('/')[-1][-9:-5])
        real_A, real_B = self.readClip(folderName, startid)
        sample = {'real_A': real_A, 'real_B': real_B, 'Name': self.list[idx]}
        return sample


class KTH_LIST_img_dyan(data.Dataset):
    def __init__(self, rootDir, N_FRAME, mode):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 64 * 64
        self.mode = mode
        if mode == 'train':
            with open('./data/e3d_kth_train_list.txt', 'r') as f:
                self.list = f.readlines()
        if mode == 'test':
            with open('./data/e3d_kth_test_list.txt', 'r') as f:
                self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def readClip(self, folderName, startid):
        path = folderName
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)

        for framenum in range(self.nfra):
            imgname_A = os.path.join(
                path, 'image_%04d' % (framenum + startid) + '.jpg')
            imgname_B = os.path.join(
                path, 'image_%04d' % (framenum + startid + 10) + '.jpg')

            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)

            img_A = cv2.resize(img_A, (64, 64))
            img_B = cv2.resize(img_B, (64, 64))

            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
            pix_A = np.array(img_A)
            pix_B = np.array(img_B)
            pix_A = pix_A / 225
            pix_B = pix_B / 225
            pix_A = pix_A[:, :, np.newaxis]
            pix_B = pix_B[:, :, np.newaxis]
            pix_A = np.transpose(pix_A, (2, 0, 1))
            pix_B = np.transpose(pix_B, (2, 0, 1))
            sample_A[:, framenum] = torch.from_numpy(pix_A.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(pix_B.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)

        path_A = folderName.split('/')[-1] + '_%05d.npy' % startid
        if self.mode == 'train':
            path_A = os.path.join(self.rootDir, 'A/train', path_A)
        if self.mode == 'test':
            path_A = os.path.join(self.rootDir, 'A/test', path_A)
        path_B = path_A.replace('/A/', '/B/')
        real_A = np.load(path_A).squeeze(0)
        real_B = np.load(path_B).squeeze(0)
        return real_A, real_B, sample_A, sample_B

    def __getitem__(self, idx):
        folderName = os.path.dirname(self.list[idx])
        startid = int(self.list[idx].split('/')[-1][-9:-5])
        real_A, real_B, frames_A, frames_B = self.readClip(folderName, startid)
        sample = {
            'real_A': real_A,
            'real_B': real_B,
            'frames_A': frames_A,
            'frames_B': frames_B,
            'Name': self.list[idx]
        }
        return sample


if __name__ == '__main__':
    rootDir = '/data/huaiyu/data/kth_action_full_exp/e3d_action_full_64'
    trainingData = torch.utils.data.DataLoader(KTH_LIST_img_dyan(
        rootDir, 10, 'train'),
        batch_size=1,
        shuffle=False)

    for i, sample in enumerate(trainingData):
        print('input real_A', sample['real_A'].shape)
        print('input real_B', sample['real_B'].shape)
        print('sample name:', sample['Name'])
        print('input frames_A', sample['frames_A'].shape)
        print('input frames_B', sample['frames_A'].shape)
        print(i)
