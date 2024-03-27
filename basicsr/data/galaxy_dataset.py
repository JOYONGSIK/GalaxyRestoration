from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, img2tensor, padding
from basicsr.utils import scandir

import random
import numpy as np
import torch
import cv2

from os import path as osp
from scipy.ndimage import gaussian_filter

class Dataset_Galaxy_Restoration(data.Dataset):
    def __init__(self, opt):
        super(Dataset_Galaxy_Restoration, self).__init__()
        print("Dataset_Galaxy_Restoration!")
        self.opt = opt
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_path = opt['dataroot_gt']
        self.lq_path = opt['dataroot_lq']

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.gt_paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        self.lq_paths = sorted(list(scandir(self.lq_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.gt_paths)

        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]
        
        img_gt = np.expand_dims(np.load(gt_path), axis=2).astype(np.float32)
        img_lq = np.expand_dims(np.load(lq_path), axis=2).astype(np.float32)
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, None)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        else:            
            np.random.seed(seed=0)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
    