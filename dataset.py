#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import os
import cv2
import PIL
import glob
import random
import numpy as np

import matplotlib.pyplot as plt


class EyeDataset(torch.utils.data.Dataset):
    """
        EyeDataset for inference.
    """

    def make_eye_dataset(self, root, train=True):
        """
            Return dataset list.
        """
        dataset = []
        if train:
            label_dir = os.path.join(root, 'train/label')
            image_dir = os.path.join(root, 'train/image')
            for fLabel in os.listdir(label_dir):
                fName = os.path.basename(fLabel)
                fImg = os.path.join(image_dir, fName)
                fLabel = os.path.join(label_dir, fName)
                dataset.append([fImg, fLabel])
        else:
            for image in os.listdir(root):
                if os.path.splitext(image)[-1] == ".jpg":
                    image_path = os.path.join(root, image)
                    dataset.append(os.path.join(root, image))
        return dataset

    def countJPG(self, root, train=True):
        """
            Count dataser number.
        """
        count = 0
        if train:
            for image in os.listdir(os.path.join(root, 'train/label')):
                if os.path.splitext(image)[-1] == ".jpg":
                    count += 1
        else:
            for image in os.listdir(root):
                if os.path.splitext(image)[-1] == ".jpg":
                    count += 1
        return count

    def __init__(self, root, train=True, transform=None):
        # check root exists
        if not os.path.exists(root):
            raise IOError("Dataset root not exist.")
        # train mode set
        self.train = train
        # crop image size
        self.nRow = 512
        self.nCol = 512
        # get val set size
        self.valSize = self.countJPG(root, self.train)
        # get path
        if self.train:
            self.train_set_path = self.make_eye_dataset(root, train)
        else:
            self.train_set_path = self.make_eye_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            # get path
            img_path, label_path = self.train_set_path[idx]
            # print("img_path: {}".format(img_path))
            # print("label_path: {}".format(label_path))
            # src image
            img = PIL.Image.open(img_path)
            img = img.resize((512, 512))
            img = np.asarray(img)
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            img = torch.from_numpy(img).float()
            img = torch.nn.functional.pad(img, (30, 30, 30, 30))
            # ground truth
            label = PIL.Image.open(label_path)
            label = label.resize((388, 388))
            label = np.asarray(label)
            label = np.atleast_3d(label).transpose(2, 0, 1)
            label = torch.from_numpy(label).float()
        else:
            # get path
            img_path = self.train_set_path[idx]
            # print("img_path: {}".format(img_path))
            # src image
            img = PIL.Image.open(img_path)
            img = img.resize((512, 512))
            img = np.asarray(img)
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            img = torch.from_numpy(img).float()
            img = torch.nn.functional.pad(img, (30, 30, 30, 30))
            # image path
            label = img_path  # label is image path here

        return img, label

    def __len__(self):
        return self.valSize


def resize_img(img, min_side=900):
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目

    return pad_img


# def resize_img(img, w_=900):
#     scale = min(float(w_) / img.shape[1], float(w_) / img.shape[0])
#     img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), cv2.INTER_LINEAR)
#     plt.imshow(img)
#     plt.show()
#     print(img.shape[0]) # h
#     print(img.shape[1]) # w
#     if img.shape[1] <= img.shape[0]:
#         pad = int((w_ - img.shape[0]) / 2)
#         pad_img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT)
#     elif img.shape[0] < img.shape[1]:
#         pad = int((w_ - img.shape[1]) / 2)
#         pad_img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT)
#     pad_img = cv2.resize(pad_img, (w_, w_), cv2.INTER_LINEAR)  # make sure out img size
#     return pad_img


def rotate(img, degree, scale=1, center=None):
    if center == None:
        center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    M = cv2.getRotationMatrix2D(center, degree, scale)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    return cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def merge(A, B, alpha):
    return cv2.addWeighted(A, 1 - alpha, B, alpha)


def merge_watermark(img, mask, alpha=0.1):
    return cv2.addWeighted(img, 1.0, mask, alpha, 0)


# def watermark(img, mask, alpha=0.3):
#     h, w = img.shape[0], img.shape[1]
#     # mask = cv2.imread(mask_path, -1)
#     # scale mask to 0.1 shape of image
#     if w > h:
#         rate = int(w * 0.1) / mask.shape[1]
#     else:
#         rate = int(h * 0.1) / mask.shape[0]
#     mask = cv2.resize(mask, None, fx=rate, fy=rate)
#     mask_h, mask_w = mask.shape[0], mask.shape[1]
#
#     mask_channels = cv2.split(mask)
#     dst_channels = cv2.split(img)
#     b, g, r, a = cv2.split(mask)
#
#     # 计算mask在图片的坐标
#     ul_points = (int(h * 0.9), int(int(w / 2) - mask_w / 2))  # 左上角
#     dr_points = (int(h * 0.9) + mask_h, int(int(w / 2) + mask_w / 2))  # 右下角
#
#     for i in range(3):
#         dst_channels[i][ul_points[0]: dr_points[0], ul_points[1]: dr_points[1]] = \
#             dst_channels[i][ul_points[0]: dr_points[0], ul_points[1]: dr_points[1]] * (255.0 - a * alpha) / 255
#         dst_channels[i][ul_points[0]: dr_points[0], ul_points[1]: dr_points[1]] += \
#             np.array(mask_channels[i] * (a * alpha / 255), dtype=np.uint8)
#     dst_img = cv2.merge(dst_channels)
#     return dst_img


class WaterMarkData(torch.utils.data.Dataset):

    def __init__(self, root, mask_root, train=0):
        self.root = root
        self.mask_root = mask_root
        self.img_list = []
        # self.mask_list = []
        self.watermark_list = []
        self.watermark_files = ['mask_1.png', 'mask_2.png', 'mask_3.png']

        for watermark_dir in self.watermark_files:
            watermark_image_dir = os.path.join(self.root, './../masks', watermark_dir)
            watermark_image = cv2.imread(watermark_image_dir, cv2.IMREAD_GRAYSCALE)
            self.watermark_list.append(watermark_image)


        file_list = os.listdir(root)
        print(file_list)
        file_list.sort()
        for f in file_list:
            if os.path.splitext(f)[-1] == '.jpg':
                try:
                    img_dir = os.path.join(self.root, f)
                    assert os.path.isfile(img_dir)
                    self.img_list.append(img_dir)
                except:
                    print("ASSERT_ERROR: File {} don't matching file.".format(f))

        print(len(self.img_list))
        self.img_num = len(self.img_list)

    def random_generate_watermark(self, shape=(900,900), random_fractor=0.05, mask_num=10):
        # Create a blank background
        background = np.zeros(shape, dtype=np.uint8)
        # Draw mask
        for i in range(mask_num):
            # Choose a mask type
            mask = random.choice(self.watermark_list)
            # Scale mask randomly
            scale_fractor = 1 - random_fractor * np.random.random()
            mask = cv2.resize(mask, (int(mask.shape[1]*scale_fractor), int(mask.shape[0]*scale_fractor)))
            # Paste mask on background
            ## choose a paste anchor randomly
            paste_anchor = (int(shape[0]*np.random.random()), int(shape[1]*np.random.random()))
            sx, sy = paste_anchor[0], paste_anchor[1]
            ex = paste_anchor[0] + mask.shape[0]
            ey = paste_anchor[1] + mask.shape[1]
            ## paste
            if ex > shape[0] or ey > shape[1]:
                continue
            background[sx:ex,sy:ey] = mask
        return background

    def add_watermark(self, img, random_fractor=0.0):
        # Step1: Generate watermark randomly
        mask = self.random_generate_watermark(random_fractor=random_fractor)
        # Step2: Scale watermark to same size with input image
        # mask = cv2.merge((mask,mask,mask))
        # Step3: Merge img & watermark together
        masked_img = merge_watermark(img, cv2.merge((mask,mask,mask)), 0.2)
        return masked_img, mask

    def item_loader(self, index):
        img = cv2.imread(self.img_list[index])
        img = resize_img(img, 900)

        img, mask = self.add_watermark(img)

        mask = cv2.resize(mask, (708,708))
        mask[mask > 0] = 1.0

        img = torch.from_numpy(img.transpose((2,0,1))).float() / 255.0
        mask = torch.from_numpy(mask).float()

        # img = img.transpose((2,0,1))

        return img, mask

    def __getitem__(self, index):
        img, mask = self.item_loader(index)
        return img, mask

    def __len__(self):
        return self.img_num



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trainset = WaterMarkData('./data/train2017', './data/masks')
    for i in range(10):
        img, mask = trainset[i]
        plt.imshow(img)
        plt.show()
        plt.imshow(mask)
        plt.show()