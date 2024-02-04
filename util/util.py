from __future__ import print_function
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from PIL import Image
import numpy as np
import os

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, label=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif label:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        # print('single-channel')
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    # print(label_tensor)  # [1,224,224]
    label_tensor = Colorize(n_label)(label_tensor)  # [3,224,224]
    label_numpy = np.transpose(label_tensor.numpy(), (2, 1, 0))  # [224,224,3] channel在后
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):  # epoch002_face_label.jpg
    # print('image_path:', image_path)
    '''
    if image_path.split('/')[-1].split('.')[0].split('_')[-1] == 'label':
        image_numpy = image_numpy[0]
    '''
    # print(image_numpy.shape)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        '''
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])
        '''
        self.cmap = np.array(
            [(0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
             (255, 255, 255), (255, 255, 255)],
            dtype=np.uint8)
        self.cmap = torch.from_numpy(self.cmap)

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(1, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            # print('label:', label, len(self.cmap))
            # print(gray_image[0], 'gray_img')
            mask = (label == gray_image[0]).cpu()
            # print(mask)
            print(self.cmap)
            print(self.cmap[label][0], 'label')
            color_image[0][mask] = self.cmap[label][0]
            # color_image[1][mask] = self.cmap[label][1]
            # color_image[2][mask] = self.cmap[label][2]

        return color_image


class Colorize_map(object):
    def __init__(self):
        self.cmap = np.array([(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
                        dtype=np.uint8)
        self.cmap = torch.from_numpy(self.cmap)

    def __call__(self, gray_image):
        size = gray_image.size()
        sketch = torch.ByteTensor(1, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):  # label=0...7
            print(gray_image[0])
            mask = (label == gray_image[0]).cpu()
            print(mask, mask.shape)  # [224, 224]
            sketch[0][mask] = 255
            # color_image[1][mask] = self.cmap[label][1]
            # color_image[2][mask] = self.cmap[label][2]

        return sketch
