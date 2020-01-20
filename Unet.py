#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=in_channels,
                                    out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=out_channels,
                                    out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels)
                )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=in_channels,
                                    out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=mid_channel,
                                    out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             output_padding=1)
                    )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=in_channels,
                                    out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=in_channels,
                                    out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=mid_channel,
                                    out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size,
                                    in_channels=mid_channel,
                                    out_channels=out_channels,
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels)
                    )
        return block

    def __init__(self, in_channel=1, out_channel=2):
        super(UNet, self).__init__()

        # Encode
        self.conv_encode1  = self.contracting_block(in_channels=in_channel,
                                                    out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2  = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3  = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode4  = self.contracting_block(256, 512)
        self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=3,
                                            in_channels=512,
                                            out_channels=1024),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(1024),
                            torch.nn.Conv2d(kernel_size=3,
                                            in_channels=1024,
                                            out_channels=1024),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(1024),
                            torch.nn.ConvTranspose2d(in_channels=1024,
                                                     out_channels=512,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1,
                                                     output_padding=1)

                )

        # Decode
        self.conv_decode4 = self.expansive_block(1024, 512, 256)
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.conv_decode1 = self.expansive_block(256, 128, 64)
        # self.final_layer  = self.final_block(128, 64, out_channels=out_channel)
        self.final_layer  = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=3,
                                    in_channels=128,
                                    out_channels=64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.Conv2d(kernel_size=3,
                                    in_channels=64,
                                    out_channels=64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.Conv2d(kernel_size=3,
                                    in_channels=64,
                                    out_channels=64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.Conv2d(kernel_size=3,
                                    in_channels=64,
                                    out_channels=out_channel,
                                    padding=2),
#                    torch.nn.ReLU()
                    torch.nn.Sigmoid()
                    )

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop: 
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            
            if (((bypass.size()[2] - upsampled.size()[2]) % 2) != 0):
                bypass = torch.nn.functional.pad(bypass, (0, -1, 0, -1))
                
            bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))
            
#        print("bypass: {}".format(bypass.shape))
#        print("upsampled: {}".format(upsampled.shape))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1  = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2  = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3  = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4  = self.conv_maxpool4(encode_block4)
        
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)

        # Decode
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer3    = self.conv_decode4(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        cat_layer2    = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1    = self.conv_decode1(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer   = self.final_layer(decode_block1)
        return final_layer


def add_conv_stage(dim_in, dim_out, kernel_size =3, stride=1, padding=1, bias=True, useBN=True):
    if useBN:
        return torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, 
                            dim_out, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding,
                            bias=bias),
            torch.nn.BatchNorm2d(dim_out),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(dim_out, 
                            dim_out, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding,
                            bias=bias),
            torch.nn.BatchNorm2d(dim_out),
            torch.nn.LeakyReLU(0.1)
            )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, 
                            dim_out, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding,
                            bias=bias),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(dim_out, 
                            dim_out, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding,
                            bias=bias),
            torch.nn.LeakyReLU(0.1)
            )

def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
    conv = torch.nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    torch.cat(conv, in_fine)
    
    
    
class UNet_new(torch.nn.Module):
    """docstring for UNet_new".Module def __init__(self, arg):
        super(UNet_new,.Module.__init__()
        self.arg = arg
    """
    pass
        
