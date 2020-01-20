#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
from torch import nn
from torchvision import transforms

import cv2
import PIL
import math
import time
import argparse
import numpy as np

from dataset import EyeDataset
from utils import *
from Unet import UNet

# ================ My Doc ==================
"""
    1. data loading need change to PIL/cv2, use lib in xundao; (finished)
    2. data size need to resize to 50*50; (finished)
    3. Combine utils for fewer code；
    4. An visulize version for compare source image and result.
    5. Test.
"""

# ------------- Parm load ------------------
parser = argparse.ArgumentParser()

parser.add_argument("--Crop_img_path", dest="Crop_img_path", type=str, 
                    default="D:\\BUCT\\Label_517\\proj\\computeye\\Sources\\plugins\\Unet\\data\\iip_19700101001242_480",
                    help="Croped image path.")
parser.add_argument("--outpath", dest="outpath", type=str,
                    help="Seg image save path.")
parser.add_argument("--checkpoint", dest="checkpoint", type=str,
                    default="./checkpoint/unet_20_0.001.pth",
                    help="Checkpoint path for net.")

args = parser.parse_args()


if __name__ == '__main__':

    # ------------ data loading ---------
    testset = EyeDataset(args.Crop_img_path, train=False, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=0)

    # ------------ init -------------
    device    = torch.device('cpu')
    unet      = UNet(in_channel=3, out_channel=1)

    # load parm
    checkpoint = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(torch.load(args.checkpoint, 
                                    map_location=device)    ['model_state_dict'])
    unet = unet.to(device)

    # ------------ infer -------------
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get data
            inputs, img_path = data
            # forward
            outputs = unet(inputs)
            # change save path
            save_path = img_path.replace("croped", "divide")
            # save img
            save_img(outputs, save_path, resize=True)