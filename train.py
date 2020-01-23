#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
from torch import nn
from torchvision import transforms

import PIL
import math
import time
import argparse
# import skimage
import numpy as np
#from visdom import Visdom

from dataset import EyeDataset, WaterMarkData
from utils import *
from Unet import UNet

#//////////////////////////////////////////////////////////////////////////////
################################# Train code ##################################
###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", dest="epoch", type=int, default=40, 
                    help="Eopch number for train.")
parser.add_argument("--lr", dest="lr", type=float, default=0.001,
                    help="learning rate for optimizer.")
parser.add_argument("--save_img", dest="save_img", type=bool, default=True,
                    help="save images to a path(True/False).")
parser.add_argument("--show_img", dest="show_img", type=bool, default=True,
                    help="show train result in training.")
parser.add_argument("--img_path", dest="img_path", type=str, default="./result",
                    help="Image save path.")
parser.add_argument("--resume", dest="resume", type=bool, default=False,
                    help="Resume training net.")
parser.add_argument("--pretrained", dest="pretrained", type=bool, default=False,
                    help="Use pretrained model.")
parser.add_argument("--val", dest="val", type=bool, default=False,
                    help="Validate mode.")
parser.add_argument("--checkpoint", dest="checkpoint", type=str, 
                    default="./checkpoint/resume.pth", help="Checkpoint load path.")

args = parser.parse_args()


if __name__ == '__main__':

    begin_timer = time.time()

    # ------- data loading ---------
    print('Loading data...')
    # set transform
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5], 
                                                 std=[0.5,0.5,0.5])
                            ])
    # # load data
    trainset = WaterMarkData('./data/train2017', './data/masks')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    # ---------- init -----------
    print('Initializing...')
    # set device
    device = torch.device("cpu")
    # init net
    unet = UNet(in_channel=3, out_channel=1)
    # load data if resume
    if args.resume:
        unet.load_state_dict(torch.load(args.checkpoint))
    # trans parm to device
    unet = unet.to(device)
    # set criterion and optimizer
    criterion = torch.nn.MSELoss()
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr, momentum=0.99)
    optimizer = torch.optim.Adagrad(unet.parameters(), lr=args.lr)
    # optimizer = torch.optim.adam(unet.parameters(), lr=args.lr)
    # trans to device
    criterion.to(device)
    # optimizer.to(device)
    
    # # init Visdom
    # viz = Visdom()
    # assert viz.check_connection()
    # temp = np.zeros((388, 388))
    # loss_counter = 0
    # src_win  = viz.image(temp,
    #                      opts=dict(title='Ground truth', caption='net output.'),
    #                      )
    # gt_win   = viz.image(temp,
    #                      opts=dict(title='Ground truth', caption='net output.'),
    #                      )
    # seg_win  = viz.image(temp,
    #                      opts=dict(title='Ground truth', caption='net output.'),
    #                      )
    # loss_win = viz.line(X=np.array([0]),
    #                     Y=np.array([0]),
    #                     opts=dict(title='loss function.')
    #                     )
    
    

    # --------- train -----------
    print('Begin training...')
    for epoch in range(args.epoch):
        print("---------- Epoch {} ----------".format(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get inputs
            inputs, seg_labels = data

            inputs = inputs.type('torch.FloatTensor')
            seg_labels = seg_labels.type('torch.FloatTensor')
            inputs = inputs.to(device)
            seg_labels = seg_labels.to(device)

            # zero gradient
            optimizer.zero_grad()
            
            # # DEBUG
            # print("input: {}".format(inputs.shape))
            # print("ground truth: {}".format(seg_labels.shape))

            # forward 
            outputs = unet(inputs)
            # outputs = active(outputs)
            
            # resize data for backward
            outputs = outputs.squeeze(1)
            seg_labels = seg_labels.squeeze(1)
            
            # # DEBUG
            # print("outsize: {}".format(outputs.shape))
            # print("seg_labels size: {}".format(seg_labels.shape))
            
            # compute loss
            loss = criterion(outputs, seg_labels)
            
            # backword
            loss.backward()
            optimizer.step()

            # print statistics
            print('[{}, {}] loss: {:.5f}'.format(epoch+1, i+1, loss.item()))
            
            # # show image
            # if args.show_img:
            #     with torch.no_grad():
            #         show_seg_result(inputs, seg_labels, outputs)
                    
            # # show image use visdom
            # with torch.no_grad():
            #     show_img_visdom(inputs, src_win, 
            #                     seg_labels, gt_win, 
            #                     outputs, seg_win, 
            #                     viz)
            #     loss_counter += 1
            #     viz.line(X=np.array([loss_counter]),
            #              Y=np.array([loss.item()]), 
            #              win=loss_win, 
            #              update='append')
                    
            # save image
            if args.save_img:
                with torch.no_grad():
                    save_path = os.path.join(args.img_path, 
                                             "{}_{}.jpg".format(epoch, i))
                    save_img(outputs, save_path)

        # if (epoch % 20 == 0):
        if True:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': unet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, './checkpoint/unet_resume_{}_{}.pth'.format(epoch, args.lr))
            torch.save(unet.state_dict(), 
                       './checkpoint/unet_{}_{}.pth'.format(epoch, args.lr))

    # print train time
    train_timer = time.time()
    time = format_time(begin_timer, train_timer)
    print('Finished Training! Using {} hour {} min {} sec.'.format(time[0], time[1], time[2]))


    # ---------- val -----------
    if args.val:
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, seg_labels = data
                outputs = unet(inputs)
    
                show_seg_result(inputs, seg_labels, outputs)
