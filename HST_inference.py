## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.utils import img2tensor
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import Denoising.utils 
from pdb import set_trace as stx
import yaml
from yaml import CLoader as Loader

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H%M%S")

np.random.seed(seed=0)  # for reproducibility

    
parser = argparse.ArgumentParser(description='Galaxy Image Restoration using Restormer')
parser.add_argument('--path_lq', default="./dataset_sample/HST/", type=str, help='Path to lq folder')
parser.add_argument('--result_dir', default='./output/HST_inference/', type=str, help='Directory for results')

parser.add_argument('--weights', default='./galaxy_pretrained_model.pth', type=str, help='Path to weights')
parser.add_argument('--config', default='./Denoising/Options/Galaxy_Restormer.yml')

args = parser.parse_args()

x = yaml.load(open(args.config, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

factor = 8
print("Model Test: Galaxy Images")
model_restoration = Restormer(**x['network_g'])    
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])

print("===>Testing using weights: ", args.weights)
print("------------------------------------------------")
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


images_gt = sorted(glob(os.path.join(args.path_lq, '*.npy')))
images_lq = sorted(glob(os.path.join(args.path_lq, '*.npy')))
        
result_dir_tmp = os.path.join(args.result_dir, f"galaxy_{current_time}")
result_dir_lq_png = os.path.join(result_dir_tmp, "lq", 'png')
result_dir_output_png = os.path.join(result_dir_tmp, "output", 'png')
result_dir_grid_png = os.path.join(result_dir_tmp, "grid", 'png')
result_dir_lq_npy = os.path.join(result_dir_tmp, "lq", 'npy')
result_dir_output_npy = os.path.join(result_dir_tmp, "output", 'npy')

os.makedirs(result_dir_tmp, exist_ok=True)
os.makedirs(result_dir_lq_png, exist_ok=True)
os.makedirs(result_dir_output_png, exist_ok=True)
os.makedirs(result_dir_grid_png, exist_ok=True)
os.makedirs(result_dir_lq_npy, exist_ok=True)
os.makedirs(result_dir_output_npy, exist_ok=True)

with torch.no_grad():
    for idx, (img_gt_path, img_lq_path) in enumerate(zip(images_gt, images_lq)):
        img_gt, img_lq = np.load(img_gt_path).astype(np.float32), np.load(img_lq_path).astype(np.float32)
        img_gt_name = img_gt_path.split('/')[-1].split('_')[0]
        img_lq_name = img_lq_path.split('/')[-1].split('_')[0]

        assert img_gt_name == img_lq_name, f"{img_gt_name} is not {img_lq_name}"
        
        if device=='cuda':
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        
        input_ = img2tensor(np.expand_dims(img_lq, axis=-1), bgr2rgb=False, float32=True).unsqueeze(0).to(device)
        
        restored = model_restoration(input_)
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # Setting path for Saving.
        save_file_lq_png_path = os.path.join(result_dir_lq_png, f"{str(idx).zfill(5)}_lq.png")
        save_file_output_png_path = os.path.join(result_dir_output_png, f"{str(idx).zfill(5)}_output.png")
        save_file_grid_png_path = os.path.join(result_dir_grid_png, f"{str(idx).zfill(5)}_grid.png")

        save_file_lq_npy_path = os.path.join(result_dir_lq_npy, f"{str(idx).zfill(5)}_lq.npy")
        save_file_output_npy_path = os.path.join(result_dir_output_npy, f"{str(idx).zfill(5)}_output.npy")

        # Saving npy images
        np.save(save_file_lq_npy_path, img_lq)
        np.save(save_file_output_npy_path, restored)
        
        # Saving png images
        restored = (restored - restored.min()) / (restored.max() - restored.min()) 
        restored = np.clip(restored, 0, 1)
        
        img_lq = (img_lq - img_lq.min()) / (img_lq.max() - img_lq.min())
        img_lq = np.clip(img_lq, 0, 1)
                
        fig, axes = plt.subplots(ncols=2)        
        axes[0].imshow(img_lq, cmap='gist_ncar')
        axes[0].set_title("Noised")
        axes[0].axis('off')
        
        axes[1].imshow(restored, cmap='gist_ncar')
        axes[1].set_title("Denoised")
        axes[1].axis('off')
        
        fig.tight_layout()
        plt.savefig(save_file_grid_png_path)
        plt.close(fig)
        plt.clf()
        
        cv2.imwrite(save_file_lq_png_path, img_as_ubyte(img_lq))
        cv2.imwrite(save_file_output_png_path, img_as_ubyte(restored))
        