## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import sys
# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
try:
    import Deshadowing.utils as utils
except ImportError:
    import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Image Deshadowing using Restormer')

parser.add_argument('--config', default=None, type=str, help='Path to yaml config file')
parser.add_argument('--input_dir', default='./Deshadowing/Datasets/', type=str, help='Directory of validation/test images')
parser.add_argument('--result_dir', default='./Deshadowing/results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Deshadowing/pretrained_models/deshadowing.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='ISTD', type=str, help='Test dataset name (e.g. ISTD, SRD) or all')
parser.add_argument('--gpus', default='0', type=str, help='CUDA device id')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

####### Load yaml #######
if args.config is not None:
    yaml_file = args.config
else:
    yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Options', 'Deshadowing_Restormer.yml')
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights, map_location='cpu')
if 'params' in checkpoint:
    model_restoration.load_state_dict(checkpoint['params'])
elif 'params_ema' in checkpoint:
    model_restoration.load_state_dict(checkpoint['params_ema'])
else:
    model_restoration.load_state_dict(checkpoint)

print(f"===> Testing using weights: {args.weights}")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 8

test_base_dir = os.path.join(args.input_dir, 'test')
if args.dataset == 'all':
    datasets = [d for d in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, d))]
else:
    datasets = [args.dataset]

for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    if os.path.exists(os.path.join(test_base_dir, dataset, 'input')):
        inp_dir = os.path.join(test_base_dir, dataset, 'input')
    elif os.path.exists(os.path.join(args.input_dir, dataset, 'input')):
        inp_dir = os.path.join(args.input_dir, dataset, 'input')
    elif os.path.exists(os.path.join(args.input_dir, 'input')):
        inp_dir = os.path.join(args.input_dir, 'input')
    else:
        inp_dir = args.input_dir

    patterns = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(inp_dir, pat)))
    files = natsorted(list(set(files)))
    
    if len(files) == 0:
        print(f"No test images found in: {inp_dir}")
        continue

    print(f"===> Processing dataset: {dataset} ({len(files)} images)")
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_)) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            filename = os.path.splitext(os.path.basename(file_))[0] + '.png'
            utils.save_img(os.path.join(result_dir, filename), img_as_ubyte(restored))

print(f"Results saved to: {args.result_dir}")
