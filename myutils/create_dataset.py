import numpy as np
from .preclassify import dicomp, hcluster
import torch
from torch.utils.data import DataLoader
from skimage import io
import math
import random


def image_normalize(data):
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def image_padding(data, r):
    if len(data.shape) == 3:
        data_new = np.lib.pad(data, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)
        return data_new
    if len(data.shape) == 2:
        data_new = np.lib.pad(data, r, 'constant', constant_values=0)
        return data_new


# 生成自然数数组并打乱
def arr(length):
    arr = np.arange(length - 1)
    # print(arr)
    random.shuffle(arr)
    # print(arr)
    return arr


# 在每个像素周围提取 patch ，然后创建成符合 pytorch 处理的格式
def createTrainingCubes(X, y, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2) + 1
    zeroPaddedX = image_padding(X, margin)
    # 把类别 uncertainty 的像素忽略
    ele_num1 = np.sum(y == 1)
    ele_num2 = np.sum(y == 2)
    patchesData_1 = np.zeros((ele_num1, patch_size, patch_size, X.shape[2]))
    patchesLabels_1 = np.zeros(ele_num1)
    patchesData_2 = np.zeros((ele_num2, patch_size, patch_size, X.shape[2]))
    patchesLabels_2 = np.zeros(ele_num2)

    patchIndex_1 = 0
    patchIndex_2 = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            # remove uncertainty pixels
            if y[r - margin, c - margin] == 1:
                patch_1 = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                patchesData_1[patchIndex_1, :, :, :] = patch_1
                patchesLabels_1[patchIndex_1] = y[r - margin, c - margin]
                patchIndex_1 = patchIndex_1 + 1
            elif y[r - margin, c - margin] == 2:
                patch_2 = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                patchesData_2[patchIndex_2, :, :, :] = patch_2
                patchesLabels_2[patchIndex_2] = y[r - margin, c - margin]
                patchIndex_2 = patchIndex_2 + 1
    patchesLabels_1 = patchesLabels_1 - 1
    patchesLabels_2 = patchesLabels_2 - 1

    # 调用arr函数打乱数组
    arr_1 = arr(len(patchesData_1))
    arr_2 = arr(len(patchesData_2))
    train_len = 8000  # 设置训练集样本数
    pdata = np.zeros((train_len, patch_size, patch_size, X.shape[2]))
    plabels = np.zeros(train_len)
    for i in range(7000):
        pdata[i, :, :, :] = patchesData_1[arr_1[i], :, :, :]
        plabels[i] = patchesLabels_1[arr_1[i]]
    for j in range(7000, train_len):
        pdata[j, :, :, :] = patchesData_2[arr_2[j - 7000], :, :, :]
        plabels[j] = patchesLabels_2[arr_2[j - 7000]]

    return pdata, plabels


def createTestingCubes(X, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2) + 1
    zeroPaddedX = image_padding(X, margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData


def getImage(args):
    if args.dataset.lower() == 'yellowriver':
        im1_path = 'SAR_data/Yellow_River_1.bmp'
        im2_path = 'SAR_data/Yellow_River_2.bmp'
        imgt_path = 'SAR_data/Yellow_River_gt.bmp'
        im1 = io.imread(im1_path).astype(np.float32)
        im2 = io.imread(im2_path).astype(np.float32)
        im_gt = io.imread(imgt_path).astype(np.float32)
    elif args.dataset.lower() == 'sulzberger':
        im1_path = 'SAR_data/Sulzberger2_1.bmp'
        im2_path = 'SAR_data/Sulzberger2_2.bmp'
        imgt_path = 'SAR_data/Sulzberger2_gt.bmp'
        im1 = io.imread(im1_path)[:, :, 0].astype(np.float32)
        im2 = io.imread(im2_path)[:, :, 0].astype(np.float32)
        im_gt = io.imread(imgt_path)[:, :, 0].astype(np.float32)
    elif args.dataset.lower() == 'chaolake':
        im1_path = 'SAR_data/Chao1_1.bmp'
        im2_path = 'SAR_data/Chao1_2.bmp'
        imgt_path = 'SAR_data/Chao1_1gt.bmp'
        im1 = io.imread(im1_path)[:, :, 0].astype(np.float32)
        im2 = io.imread(im2_path)[:, :, 0].astype(np.float32)
        im_gt = io.imread(imgt_path)[:, :, 0].astype(np.float32)

    else:
        raise ValueError('Unknown dataset')

    im_di = dicomp(im1, im2)
    ylen, xlen = im_di.shape
    pix_vec = im_di.reshape([ylen * xlen, 1])

    preclassify_lab = hcluster(pix_vec, im_di)
    print('... ... hiearchical clustering finished !!!')
    patch_size = args.patchsize
    return im1, im2, im_gt, im_di, patch_size, preclassify_lab


def createDataset(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab):
    ylen, xlen = im_di.shape
    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:, :, 0] = im1
    mdata[:, :, 1] = im2
    mdata[:, :, 2] = im_di
    mlabel = preclassify_lab

    x_train, y_train = createTrainingCubes(mdata, mlabel, patch_size)
    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = createTestingCubes(mdata, patch_size)
    x_test = x_test.transpose(0, 3, 1, 2)

    return x_train, y_train, x_test


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, args, im1, im2, im_gt, im_di, patch_size, preclassify_lab):
        x_train, y_train, _ = createDataset(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab)
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def get_train_loader(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab) -> torch.utils.data.DataLoader:
    train_set = TrainDS(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=True,
                                               num_workers=0)
    return train_loader
