import os

rootDir = '/data/Huaiyu/DYAN/data/kth/processed'
listOfFolders = []
classesList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
classesList.sort()

for i in range(len(classesList)):
    classes_videoList = []
    classesList[i] = os.path.join(rootDir, classesList[i])
    classes_videoList = [name for name in os.listdir(classesList[i]) if
                         os.path.isdir(os.path.join(classesList[i]))]
    classes_videoList.sort()

    for j in range(len(classes_videoList)):
        classes_videoList[j] = os.path.join(classesList[i], classes_videoList[j])
        print(classes_videoList[j])
    listOfFolders.extend(classes_videoList)
listOfFolders.sort()

sum = 0
list = []
for i in range(len(listOfFolders)):
    frames = [each for each in os.listdir(listOfFolders[i]) if each.endswith('.png')]
    nFrames = len(frames)
    # print(nFrames)
    nClips = (nFrames // 20) - 1
    sum = sum + nClips
    for j in range(0, nClips):
        startid = j * 20
        startid_path = os.path.join(listOfFolders[i], '%05d' % startid)
        # print(startid_path)
        list.append(startid_path)
        # print(j, startid_path)
with open('./your_file.txt', 'w') as f:
    for item in list:
        f.write("%s\n" % item)
