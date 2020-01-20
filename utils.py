# -*- coding: utf-8 -*-

import torch
import torchvision

import os
import PIL
import cv2
# import skimage
import numpy as np
# import matplotlib.pyplot as plt

# from visdom import Visdom

def format_time(begin_time, end_time):
    # return a tuple with (hour, minutes, secs)
    total_time = begin_time - end_time
    hour    = total_time // 360
    minutes = (total_time - (hour*360)) // 60
    secs    = total_time % 60
    time = (hour, minutes, secs)
    return time

def show_seg_result(image, label, seg_label):
    """show (image, label, ground_truth)"""
    # prepare img
    npimage = np.array(torchvision.utils.make_grid(image))
    # npimage /= 255  # normailze value

    nplabel = np.array((torchvision.utils.make_grid(label)))
    # nplabel /= 255  # normailze value
    
    npseg_label = np.array(torchvision.utils.make_grid(seg_label.detach()))
    npseg_label = npseg_label > 0.5
    npseg_label = np.uint8(npseg_label*255)
    # npseg_label /= 255  # normailze value

    # show image
    print('Image: {}'.format(npimage.shape))
    # # use plt
    # plt.imshow(np.transpose(npimage, (1,2,0)), cmap='Accent')
    # plt.show()
    # use cv2
    img = cv2.fromarray(npimage)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img) 
    cv2.waitKey (0)

    print('Ground truth: {}'.format(nplabel.shape))
    # # use plt
    # plt.imshow(np.transpose(nplabel, (1,2,0)), cmap='Greys')
    # plt.show()
    # use cv2
    label = cv2.fromarray(nplabel)
    cv2.namedWindow("Ground_truth")
    cv2.imshow("Ground_truth", label) 
    cv2.waitKey (0)
    
    print('Seg result: {}'.format(npseg_label.shape))
    # # use plt
    # plt.imshow(np.transpose(npseg_label, (1,2,0)), cmap='Accent')
    # plt.show()
    # use cv2
    seg_label = cv2.fromarray(npseg_label)
    cv2.namedWindow("Seg result")
    cv2.imshow("Seg result", seg_label) 
    cv2.waitKey (0)

    cv2.destroyAllWindows()
    

# def show_img_visdom(image, img_win, label, gt_win, seg_label, seg_win, viz):
#     """show (image, label, ground_truth) in visdom"""
#     # prepare img
#     image   = torch.nn.functional.pad(image, (-30, -30, -30, -30))
#     npimage = np.array(torchvision.utils.make_grid(image).cpu())
# #    npimage = np.uint8(npimage)
# #    print("npimage: {}".format(type(npimage)))
# #    image_pl = PIL.Image.fromarray(npimage)
# #    image_pl.resize(388, 388)
# #    npimage = np.array(image_pl)

#     nplabel = np.array(torchvision.utils.make_grid(label).cpu())
    
#     npseg_label = np.array(torchvision.utils.make_grid(seg_label.detach()).cpu())
#     npseg_label = npseg_label > 0.5
#     npseg_label = np.uint8(npseg_label*255)

#     # show image
#     viz.image(npimage, win=img_win)
#     viz.image(nplabel, win=gt_win)
#     viz.image(npseg_label, win=seg_win)
    
# no test
def save_img(img, path, resize=False, img_shape=(50, 50)):
    """ save image """
    # check path exist
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # save image
    img = np.array(torchvision.utils.make_grid(img).cpu())
    img = img[0,:,:]
    img = img > 0.5
    img = PIL.Image.fromarray(np.uint8(img*255), mode='L')
    if resize:
        img = img.resize(img_shape)
    img.save(path)
    
    
if __name__ == '__main__':
    image = PIL.Image.open('C:\\work\\unet_seg\\data\\membrane\\test\\0.png')
    label = PIL.Image.open('C:\\work\\unet_seg\\data\\membrane\\test\\0_predict.png')
    seg_label = PIL.Image.open('C:\\work\\unet_seg\\data\\membrane\\test\\0_predict.png')
    show_seg_result(image, label, seg_label)