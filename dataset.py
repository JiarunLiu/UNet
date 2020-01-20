#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import os
import PIL
import glob
import numpy as np


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
                fName  = os.path.basename(fLabel)
                fImg   = os.path.join(image_dir, fName)
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
            img   = PIL.Image.open(img_path)
            img   = img.resize((512, 512))
            img   = np.asarray(img)
            img   = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img   = (img - img.min()) / (img.max() - img.min())  # Normalize
            img   = torch.from_numpy(img).float()
            img   = torch.nn.functional.pad(img, (30, 30, 30, 30))
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
            img   = PIL.Image.open(img_path)
            img   = img.resize((512, 512))
            img   = np.asarray(img)
            img   = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img   = (img - img.min()) / (img.max() - img.min())  # Normalize
            img   = torch.from_numpy(img).float()
            img   = torch.nn.functional.pad(img, (30, 30, 30, 30))
            # image path
            label = img_path  # label is image path here
            
        return img, label

    def __len__(self):
        return self.valSize