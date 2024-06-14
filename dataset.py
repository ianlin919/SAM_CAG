""" train and test dataset

author jundewu
"""
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset
from utils import random_click, random_box, loadTxt
from RandAugment import *

class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'none', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
        multi_rater_disc = [Image.open(path).convert('L') for path in multi_rater_disc_path]

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        multi_rater_cup_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_cup]
        multi_rater_disc_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_disc]

        # first click is the target agreement among most raters
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(np.mean(np.stack(multi_rater_cup_np), axis=0)) / 255, point_label)
            point_label, pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label)
        else:
            # you may want to get rid of click prompts
            pt = np.array([0, 0], dtype=np.int32)
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
            # transform to mask size (out_size) for mask define
            mask_cup = F.interpolate(multi_rater_cup, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            multi_rater_disc = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)
            mask_disc = F.interpolate(multi_rater_disc, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)
            torch.set_rng_state(state)
            
            mask = torch.concat([mask_cup, mask_disc], dim=0)

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box': box_cup,
            'image_meta_dict':image_meta_dict,
        }
    

class LIDC(Dataset):
    names = []
    images = []
    labels = []
    series_uid = []

    def __init__(self, data_path, transform=None, transform_msk = None, prompt = 'click'):
        self.prompt = prompt
        self.transform = transform
        self.transform_msk = transform_msk
        
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
                
        
        for key, value in data.items():
            self.names.append(key)
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        point_label = 1

        """Get the images"""
        img = np.expand_dims(self.images[index], axis=0)
        name = self.names[index]
        multi_rater = self.labels[index]

        # first click is the target most agreement among raters, otherwise, background agreement
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(np.mean(np.stack(multi_rater), axis=0)) / 255, point_label)

        # Convert image (ensure three channels) and multi-rater labels to torch tensors
        img = torch.from_numpy(img).type(torch.float32)
        img = img.repeat(3, 1, 1) 
        multi_rater = [torch.from_numpy(single_rater).type(torch.float32) for single_rater in multi_rater]

        multi_rater = torch.stack(multi_rater, dim=0)
        multi_rater = multi_rater.unsqueeze(1)

        if self.prompt == 'box':
            x_min, x_max, y_min, y_max = random_box(multi_rater)
            box = [x_min, x_max, y_min, y_max]

        mask = multi_rater.mean(dim=0) # average

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box': box,
            'image_meta_dict':image_meta_dict,
        }
        
class CAG(Dataset):
    def __init__(self, args, data_path, txt_path, transform=None, transform_msk=None, mode='Training', prompt='click',
                 plane=False, selected_name=None):

        fileNames = [name for name in loadTxt(str(txt_path))]
        self.fileNames = fileNames
        # all_name = [name for name in os.listdir(f'{data_path}/imgs')]
        # # df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        # # all_name = [f'CVAI-0005_LCX_RAO25_CAU35_39.png']
        data_path = f'/nmt/workspace/CAG/train/5F_2'

        # if selected_name is not None:
        #     self.name_list = [f'{data_path}/imgs/{name}' for name in selected_name]
        #     self.label_list = [f'{data_path}/labels/{name}' for name in selected_name]
        # else:
        #     self.name_list = [f'{data_path}/imgs/{name}' for name in all_name]
        #     self.label_list = [f'{data_path}/labels/{name}' for name in all_name]

        # self.name_list = df.iloc[:, 1].tolist()
        # self.label_list = df.iloc[:, 2].tolist()
        
        self.data_path = data_path
        self.img_dir = os.path.join(self.data_path , 'imgs')
        self.label_dir = os.path.join(self.data_path ,'labels')
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.fileNames)


    def __getitem__(self, idx):
        inout = 1
        point_label = 1

        """Get the images"""
        # name = self.name_list[index]
        # img_path = os.path.join(self.data_path, name)
        #
        # mask_name = self.label_list[index]
        # msk_path = os.path.join(self.data_path, mask_name)
        fileName = self.fileNames[idx]
        img = Image.open(os.path.join(self.img_dir, fileName)).convert('RGB')
        mask = cv2.imread(os.path.join(self.label_dir, fileName), flags=cv2.IMREAD_UNCHANGED)
        # img = Image.open(img_path).convert('RGB')
        # mask = Image.open(msk_path).convert('L')
        # print(mask_name)
        # mask = cv2.imread(mask_name, flags=cv2.IMREAD_UNCHANGED)
        mask = 255 * (mask.astype(np.float32) / mask.max())
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        newsize = (self.img_size, self.img_size)
        # mask = cv2.resize(mask, newsize)

        # if self.prompt == 'click':
        #     ic(np.array(mask).shape)
        #     pt = random_click(np.array(mask) / np.array(mask).max(), point_label, inout)


        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label,inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
        fileName = fileName.split('/')[-1].split(".")[0]
        image_meta_dict = {'filename_or_obj': fileName}
        # ic(pt.shape)
        # ic(p_label.shape)
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }