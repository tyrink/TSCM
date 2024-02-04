import os
import cv2
import sys
import glob
import tqdm
import time
import torch
import random
import argparse
import fractions
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


from cr_json import get_face_label
cv2.setNumThreads(0)


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def encode_segmentation_rgb(segmentation, face_mask=True):
    parse = segmentation[:,:,0]

    face_part_ids = [1, 6, 7, 4, 5, 3, 2, 10, 11, 12, 13] if face_mask else [4, 5, 3, 2, 10, 11, 12, 13]

    face_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse == valid_id)
        face_map[valid_index] = 255

    return face_map


transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize([-0.5, -0.5, -0.5], [1, 1, 1])
    ])

transformer_seg = transforms.Compose([
            transforms.Resize(224, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # follow ImageNet
        ])

if __name__ == '__main__':
    opt = TestOptions().parse()

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    test_path = '../testdata/img_params'
    file_list = glob.glob(test_path + '/*')
    file_list.sort()

    for i in range(len(file_list)):
            p1 = random.randint(0, len(file_list) - 1)
            p2 = p1
            while p2 == p1:
                p2 = random.randint(0, len(file_list) - 1)

            src_param = file_list[p1]
            tat_param = file_list[p2]

            pic_a = src_param.replace('3dparam/faces', 'faces')
            pic_a = pic_a.replace('npy', 'jpg')
            pic_b = tat_param.replace('3dparam/faces', 'faces')
            pic_b = pic_b.replace('npy', 'jpg')

            src_n = pic_a.split('/')[-1].split('.')[0]
            tat_n = pic_b.split('/')[-1].split('.')[0]

            latent_a = src_param.replace('3dparam/faces', 'id_arc')
            dd_a = src_param  
            latent_b = tat_param.replace('3dparam/faces', 'id_arc')
            dd_b = tat_param  
            box_b = dd_b.replace('3dparam', '3dbox')

            seg_tat_path = pic_b.replace('faces', 'parsing_map')
            seg_tat_path = seg_tat_path.replace('jpg', 'png')
            st_src = src_param.replace('3dparam', 'st_codes')
            st_tat = tat_param.replace('3dparam', 'st_codes')

            with torch.no_grad():
                # pic_a = opt.pic_a_path
                start_time = time.time()
                seg_tat = Image.open(seg_tat_path)
                # print(seg_tat.size)
                seg_tat = transformer_seg(seg_tat) * 255.0
                seg_tat[seg_tat == 255] = 19
                seg_tat = seg_tat.unsqueeze_(0).cuda()

                seg_att_ = cv2.imread(seg_tat_path)
                t_mask = encode_segmentation_rgb(seg_att_)
                organ_mask = encode_segmentation_rgb(seg_att_, face_mask=False)
                t_mask = cv2.resize(t_mask, (224, 224))
                organ_mask = cv2.resize(organ_mask, (224, 224))
                t_mask = t_mask.astype(np.float) / 255.0
                organ_mask = organ_mask.astype(np.float) / 255.0
                t_mask_blur = cv2.GaussianBlur(t_mask, (51, 51), 0)

                t_mask_blur = torch.from_numpy(t_mask_blur).unsqueeze_(0).cuda()
                organ_mask = torch.from_numpy(organ_mask).unsqueeze_(0).cuda()

                img_a = Image.open(pic_a).convert('RGB').resize((224, 224), Image.BILINEAR)
                img_ao = transformer(img_a)
                img_a = transformer_Arcface(img_a)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

                # pic_b = opt.pic_b_path
                img_b = Image.open(pic_b).convert('RGB').resize((224, 224), Image.BILINEAR)
                img_bo = transformer(img_b)
                img_b = transformer_Arcface(img_b)
                img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

                pic_name = '%s_%s_pic' % (src_n, tat_n)

                # convert numpy to tensor
                img_ao = img_ao.cuda()
                img_bo = img_bo.cuda()
                img_id = img_id.cuda()
                img_att = img_att.cuda()

                latent_id = np.load(latent_a)
                latent_id = latent_id / np.linalg.norm(latent_id)

                latent_att = np.load(latent_b)
                latent_att = latent_att / np.linalg.norm(latent_att)
                latent_att = torch.from_numpy(latent_att).cuda()
                # print(img_att.device, latent_att.device, latent_id.device)

                st_src = np.load(st_src).squeeze(0)
                st_src = st_src / np.linalg.norm(st_src)
                st_src = torch.from_numpy(st_src).cuda()
                st_tat = np.load(st_tat).squeeze(0)
                st_tat = st_tat / np.linalg.norm(st_tat)
                st_tat = torch.from_numpy(st_tat).cuda()

                dd_id = np.load(dd_a)
                # dd_id = dd_id / np.linalg.norm(dd_id)
                dd_att = np.load(dd_b)
                # dd_att = dd_att / np.linalg.norm(dd_att)
                dd_couple = np.hstack((dd_att[0:12], dd_id[12:52]))
                dd_couple = np.hstack((dd_couple, dd_att[52:62]))
                dd_param = [dd_couple]
                box_att = np.load(box_b)
                roi_box_list = [box_att.tolist()]
                face_label_pic_o = get_face_label(roi_box_list, dd_param)
                face_label_pic = transformer(face_label_pic_o.astype(np.float32)).unsqueeze(0).cuda()

                latent_id = torch.from_numpy(latent_id).cuda()
                ############## Forward Pass ######################
                img_fake = model(img_id, img_att, latent_id, latent_att, face_label_pic, None, True, st_src, st_tat, seg_tat, t_mask_blur, organ_mask)  #
                end_time = time.time()
              
                full = torch.cat([img_ao, img_bo, face_label_pic[0], img_fake[0]], dim=2).detach()
                full0 = img_fake[0].detach()
                # full0 = detransformer(full0)
                # print(full.shape)
                full = full.permute(1, 2, 0)  # H,W,C
                output = full.to('cpu')
                output = np.array(output)
                output = output[..., ::-1]  # inverse, bgr->rgb
                output = output * 255
                full0 = full0.permute(1, 2, 0)  # H,W,C
                output0 = full0.to('cpu')
                output0 = np.array(output0)
                output0 = output0[..., ::-1]  # inverse, bgr->rgb
                output0 = output0 * 255
                os.makedirs(opt.output_path, exist_ok=True)
                cv2.imwrite(os.path.join(opt.output_path, '%s_%s_img.jpg' % (src_n, tat_n)), output0)


