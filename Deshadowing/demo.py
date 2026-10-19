"""
Restormer Deshadowing - Simple Inference Demo Script
Quickly test deshadowing on a single image or an entire folder of images.

Usage:
    # 1. Single image:
    python3 Deshadowing/demo.py --input path/to/shadow_doc.jpg --output output.png

    # 2. Entire directory of images:
    python3 Deshadowing/demo.py --input path/to/folder/ --output path/to/results/
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from basicsr.models.archs.restormer_arch import Restormer

def load_model(weights_path, device='cuda'):
    print(f"[*] Loading Restormer model from: {weights_path}")
    # Architecture matches trained High-Frequency Aware Restormer
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
        use_hf_branch=True
    )

    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def process_image(model, img_path, device='cuda', factor=8):
    # Read image using PIL / OpenCV in RGB [0, 1]
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # [H, W, C] -> [1, C, H, W]
    input_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad image to multiple of factor (8) for transformer windowing
    h, w = input_t.shape[2], input_t.shape[3]
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    padh = H - h
    padw = W - w
    input_padded = F.pad(input_t, (0, padw, 0, padh), mode='reflect')

    with torch.no_grad():
        restored = model(input_padded)

    # Unpad to original size
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0.0, 1.0)
    out_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_uint8 = np.round(out_np * 255.0).astype(np.uint8)
    return out_uint8

def main():
    parser = argparse.ArgumentParser(description="Restormer Deshadowing Inference Demo")
    parser.add_argument("--input", required=True, type=str, help="Path to degraded image file or folder")
    parser.add_argument("--output", default="./Deshadowing/results/demo", type=str, help="Output image file or directory")
    parser.add_argument("--weights", default="./experiments/Deshadowing_Restormer_RDD/models/net_g_latest.pth", type=str, help="Path to .pth checkpoint file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device (cuda or cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        # Fallback to 100k checkpoint if latest is not found
        alt_weights = "./experiments/Deshadowing_Restormer_RDD/models/net_g_100000.pth"
        if os.path.exists(alt_weights):
            args.weights = alt_weights
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {args.weights}")

    model = load_model(args.weights, device=args.device)

    # Check if input is a single file or directory
    if os.path.isfile(args.input):
        print(f"[*] Processing single image: {args.input}")
        out_img = process_image(model, args.input, device=args.device)
        
        # If output ends with image extension, save directly; otherwise save into directory
        if os.path.splitext(args.output)[1].lower() in ['.png', '.jpg', '.jpeg']:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            save_path = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            basename = os.path.splitext(os.path.basename(args.input))[0] + "_deshadowed.png"
            save_path = os.path.join(args.output, basename)

        Image.fromarray(out_img).save(save_path)
        print(f"[✓] Saved restored image to: {save_path}")

    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.BMP']
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(args.input, ext)))
        files = sorted(list(set(files)))

        if len(files) == 0:
            print(f"[!] No image files found in: {args.input}")
            return

        print(f"[*] Found {len(files)} images in {args.input}. Running inference...")
        for f in tqdm(files):
            out_img = process_image(model, f, device=args.device)
            save_name = os.path.splitext(os.path.basename(f))[0] + ".png"
            Image.fromarray(out_img).save(os.path.join(args.output, save_name))
        print(f"[✓] All {len(files)} images processed and saved to: {args.output}")
    else:
        print(f"[!] Error: input path does not exist: {args.input}")

if __name__ == "__main__":
    main()
