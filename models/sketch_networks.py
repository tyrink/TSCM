"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import cv2
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import math
import datetime
from torchvision import utils
from pathlib import Path

ddfa_dir = Path('/home1/lfy/try/code_sims/SimSwap-main/data/DDFA_V2/')
sys.path.append(str(ddfa_dir))
from parsing_model.model import BiSeNet


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle_id(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels):
        super(ApplyStyle_id, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        # global id embedding
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        # x_id = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        x_id = x * (style[:, 0] * 1) + style[:, 1] * 1
        return x_id


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(False)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle_id(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle_id(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class ApplyStyle_mix(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, channels):
        super(ApplyStyle_mix, self).__init__()
        # nhidden = 128
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        # style
        self.style_length = 512
        self.conv_gamma = nn.Conv2d(self.style_length, channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(self.style_length, channels, kernel_size=3, padding=1)

        # pic_sp
        '''
        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        '''
        self.conv_weight = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv_bias = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # blending_param
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, face_label_pic, z_st):
        x = self.norm(x)
        # local_style_enhance & structure control
        z_st = F.interpolate(z_st, x.size(2), mode='bicubic', align_corners=True)
        gamma_st = self.conv_gamma(z_st)
        beta_st = self.conv_beta(z_st)

        # pic_sp_params
        face_label_pic = F.interpolate(face_label_pic, x.size(2), mode='bicubic', align_corners=True)
        # acti = self.conv_shared(face_label_pic)
        w_pic = self.conv_weight(face_label_pic)
        b_pic = self.conv_bias(face_label_pic)

        # param_blending
        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)
        gamma_final = gamma_alpha * gamma_st + (1 - gamma_alpha) * w_pic
        beta_final = beta_alpha * beta_st + (1 - beta_alpha) * b_pic
        x_id = x * (1 + gamma_final) + beta_final

        return x_id


class ResnetBlock_ada(nn.Module):
    def __init__(self, dim_in, dim_out, padding_type, activation=nn.ReLU(False)):
        super(ResnetBlock_ada, self).__init__()
        self.learned_shortcut = (dim_in != dim_out)

        if self.learned_shortcut:
            self.style_s = ApplyStyle_mix(dim_in)
            self.act_s = activation
            self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim_in*2, dim_out, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle_mix(dim_in)
        self.act1 = activation

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim_out*2, dim_out, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle_mix(dim_out)
        self.act2 = activation

        conv3 = []
        if padding_type == 'reflect':
            conv3 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv3 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv3 += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p), InstanceNorm()]
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, x, face_label_pic, z_st, feat_id):
        # x_s = self.style_s(x, face_label_pic, z_st) 
        # x_s = self.act_s(x_s)
        feat_id_ = self.conv3(feat_id)
        x_s = self.conv_s(x)
        y = self.style1(x, face_label_pic, z_st)
        y_c1 = torch.cat([y, feat_id], dim=1)
        y = self.act1(y_c1)
        y = self.conv1(y) # with id
        y = self.style2(y, face_label_pic, z_st)
        y_c2 = torch.cat([y, feat_id_], dim=1)
        y = self.act2(y_c2)
        y = self.conv2(y)
        out = x_s + y

        return out
    
class OIT(nn.Module):
    def __init__(self, dim_in, dim_out, latent_size=512, n_blocks=6, padding_type='reflect'):
        super(OIT, self).__init__()
        
        self.linear = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out*2)
        self.activation = nn.ReLU(False)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(dim_out, latent_size=latent_size, padding_type=padding_type, activation=self.activation)]
        self.BottleNeck = nn.Sequential(*BN)

    def forward(self, tat_fea, latent_att, latent_id):
        b, c, h, w = tat_fea.shape
        q_id = self.linear(latent_att)  # [8,1,dim_out]
        kv_fea = self.linear2(tat_fea.flatten(start_dim=2, end_dim=3).transpose(-2, -1)).reshape(-1, h * w, c, 2).permute(0, 3, 1, 2)
        key_fea = kv_fea[:, 0]
        atten_fea = (q_id @ key_fea.transpose(-2, -1)) / math.sqrt(c)  # [8,1,784]
        atten_fea = atten_fea.softmax(dim=-1).reshape(-1, 1, h, w)
        fea_all = kv_fea[:, 1].transpose(-2, -1).reshape(-1, c, h, w)
        tat_fea_id_re = fea_all * atten_fea  # tat_fea * atten_fea
        tat_fea_id_unre = fea_all * (torch.ones_like(atten_fea).to(atten_fea.device) - atten_fea)  # tat_fea - fea_id_re
    
        # id-embedding adain
        for i in range(len(self.BottleNeck)): # 4*adain
            tat_fea_id_re = self.BottleNeck[i](tat_fea_id_re, latent_id)

        x = tat_fea_id_unre + tat_fea_id_re

        return x


class Encoder(nn.Module):
    def __init__(self, input_nc, deep=False, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        activation = nn.ReLU(False)
        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
        self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)

    def forward(self, x):  # 3-channel input
        x = self.first_layer(x)
        y1 = self.down1(x)
        y2 = self.down2(y1)
        out = self.down3(y2)
        if self.deep:
            out = self.down4(out)

        return out, y2, y1



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        y = self.fc(avg_out).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()
        activation = nn.ReLU(False)
        self.deep = deep

        self.encoder_tat = Encoder(input_nc, deep=False, norm_layer=nn.BatchNorm2d)
        self.oit_1 = OIT(512, 128, latent_size=512, n_blocks=4)
        self.oit_2 = OIT(512, 256, latent_size=512, n_blocks=4)
        self.oit_3 = OIT(512, 512, latent_size=512, n_blocks=4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        if self.deep:
            self.up4 = ResnetBlock_ada(512, 512, padding_type=padding_type, activation=activation)
        self.up3 = ResnetBlock_ada(512, 256, padding_type=padding_type, activation=activation)
        self.up2 = ResnetBlock_ada(256, 128, padding_type=padding_type, activation=activation)
        # self.up1 = ResnetBlock_ada(128, 64, padding_type=padding_type, activation=activation)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                                        nn.Tanh())

        self.se_atten_3 = ChannelAttention(512)
        self.se_atten_2 = ChannelAttention(256)

        # self.conv_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_merge2 = nn.Conv2d(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_merge3 = nn.Conv2d(512 + 256, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, img_tat, latent_id, latent_att, face_label_pic, st_src, st_tat, seg_tat, face_mask, organ_mask):
        tat_fea, tat2, tat1 = self.encoder_tat(img_tat) # 28, 56, 112
        
        feat_id_3 = self.oit_3(tat_fea, latent_att, latent_id) # 512 28
        feat_id_2 = self.oit_2(tat2, latent_att, latent_id) # 256 56

        # cross-scale interaction
        # 3->2 upscale
        feat_id_3_up = F.interpolate(feat_id_3, scale_factor=2, mode='bicubic')
        feat_id_3_up = F.relu(self.conv_3(feat_id_3_up))
        feat_id_2_concat = torch.cat((feat_id_3_up, feat_id_2), dim=1)
        feat_id_2_ = F.relu(self.conv_merge2(feat_id_2_concat))
        # print('feat_id_2', feat_id_2_concat.shape, feat_id_2_.shape)

        # 2->3 downscale
        feat_id_2_down = F.relu(self.conv3_2(feat_id_2))
        feat_id_3_concat = torch.cat((feat_id_2_down, feat_id_3), dim=1)
        feat_id_3_ = F.relu(self.conv_merge3(feat_id_3_concat))
        # print('feat_id_3', feat_id_3_concat.shape, feat_id_3_.shape)

        feat_id_3_se = self.se_atten_3(feat_id_3_)
        feat_id_2_se = self.se_atten_2(feat_id_2_)
     
        face_mask = face_mask.unsqueeze_(1).float()
        organ_mask = organ_mask.unsqueeze_(1).float()

        bs, _, h, w = seg_tat.size()  # [8,1,512,512]
        nc = 19

        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, seg_tat.long(), 1.0)  # dim=1   [8,19,512,512]

        tat_mask = F.interpolate(input_semantics, tat_fea.size(2), mode='bilinear')
        organ_mask = F.interpolate(organ_mask, tat_fea.size(2), mode='bilinear')

        tat_mask_ = tat_mask.flatten(start_dim=2, end_dim=3)

        z_st_tat = (st_tat.transpose(-2, -1) @ tat_mask_).reshape(-1, 512, 28, 28)  # proved to be right
        z_st_src = (st_src.transpose(-2, -1) @ tat_mask_).reshape(-1, 512, 28, 28)

        # new_version
        organ_mask_ = organ_mask.to(torch.bool).expand([bs, 512, -1, -1])
        z_st_tat[organ_mask_] = z_st_src[organ_mask_]
        z_st = z_st_tat

        x = tat_fea
        if self.deep:
            x = self.up(x)
            x = self.up4(x, face_label_pic, z_st)
        x = self.up(x)
        x = self.up3(x, face_label_pic, z_st, self.up(feat_id_3_se))
        x = self.up(x)
        x = self.up2(x, face_label_pic, z_st, self.up(feat_id_2_se))
        '''
        # fusion_with_bkg
        face_mask_normed = F.interpolate(face_mask, x_.size(2), mode='bilinear')
        # fusion_tat = F.interpolate(fusion_tat, x_.size(2), mode='bilinear')
        x = x_ * face_mask_normed + fusion_tat * (torch.ones_like(face_mask_normed) - face_mask_normed)
        '''
        # x = self.up(x)
        # x = self.up1(x, face_label_pic, z_st)
        x = self.up1(x)
        x = self.last_layer(x)
        x = (x + 1) / 2

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128), nn.LeakyReLU(0.2, True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256), nn.LeakyReLU(0.2, True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
            norm_layer(512), nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

    def forward(self, input):
        out = []
        x = self.down1(input)
        out.append(x)
        x = self.down2(x)
        out.append(x)
        x = self.down3(x)
        out.append(x)
        x = self.down4(x)
        out.append(x)
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        out.append(x)

        return out