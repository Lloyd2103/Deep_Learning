## Restormer Deshadowing Evaluation Script (Python)
## Calculates PSNR, SSIM, and RMSE (All, Shadow, Non-Shadow)

import os
import argparse
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def calculate_rmse(img1, img2, mask=None):
    """
    Calculate RMSE between img1 and img2 in LAB space (standard in deshadowing benchmarks).
    img1, img2: RGB uint8 in [0, 255]
    mask: binary mask uint8 [0, 255], 255 for shadow, 0 for non-shadow.
    """
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float64)
    diff = (lab1 - lab2) ** 2
    diff = np.sum(diff, axis=2) # sum across L, A, B channels

    if mask is None:
        return np.sqrt(np.mean(diff))
    
    mask_bool = mask > 127
    if np.sum(mask_bool) == 0:
        return 0.0
    return np.sqrt(np.mean(diff[mask_bool]))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deshadowing Results")
    parser.add_argument("--res_dir", default="./Deshadowing/results/ISTD", type=str, help="Directory containing restored images")
    parser.add_argument("--gt_dir", default="./Deshadowing/Datasets/test/ISTD/target", type=str, help="Directory containing ground-truth images")
    parser.add_argument("--mask_dir", default="./Deshadowing/Datasets/test/ISTD/mask", type=str, help="Directory containing shadow mask images (optional)")
    args = parser.parse_args()

    files = natsorted(glob(os.path.join(args.res_dir, "*.png")) + glob(os.path.join(args.res_dir, "*.jpg")))
    if len(files) == 0:
        print(f"No result images found in: {args.res_dir}")
        return

    has_mask = os.path.exists(args.mask_dir) and len(os.listdir(args.mask_dir)) > 0

    total_psnr = 0.0
    total_ssim = 0.0
    rmse_all_list = []
    rmse_shadow_list = []
    rmse_non_shadow_list = []

    print(f"===> Evaluating {len(files)} images from: {args.res_dir}")
    print(f"Ground Truth: {args.gt_dir}")
    if has_mask:
        print(f"Mask: {args.mask_dir}")
    else:
        print("Mask: Not found (calculating Overall PSNR, SSIM, and RMSE only)")

    for f in tqdm(files):
        fname = os.path.basename(f)
        basename, _ = os.path.splitext(fname)

        res_img = cv2.imread(f)
        if res_img is None:
            continue
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

        # Look for matching GT file
        gt_candidates = glob(os.path.join(args.gt_dir, basename + ".*"))
        if len(gt_candidates) == 0:
            print(f"Warning: GT not found for {fname}")
            continue
        gt_img = cv2.imread(gt_candidates[0])
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        if res_img.shape != gt_img.shape:
            res_img = cv2.resize(res_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_CUBIC)

        cur_psnr = compute_psnr(gt_img, res_img, data_range=255)
        cur_ssim = compute_ssim(gt_img, res_img, channel_axis=2, data_range=255)
        total_psnr += cur_psnr
        total_ssim += cur_ssim

        cur_rmse_all = calculate_rmse(gt_img, res_img)
        rmse_all_list.append(cur_rmse_all)

        if has_mask:
            mask_candidates = glob(os.path.join(args.mask_dir, basename + ".*"))
            if len(mask_candidates) > 0:
                mask_img = cv2.imread(mask_candidates[0], cv2.IMREAD_GRAYSCALE)
                if mask_img.shape != gt_img.shape[:2]:
                    mask_img = cv2.resize(mask_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                cur_rmse_shadow = calculate_rmse(gt_img, res_img, mask=mask_img)
                cur_rmse_non_shadow = calculate_rmse(gt_img, res_img, mask=(255 - mask_img))
                rmse_shadow_list.append(cur_rmse_shadow)
                rmse_non_shadow_list.append(cur_rmse_non_shadow)

    n = len(rmse_all_list)
    print("\n" + "=" * 45)
    print("           DESHADOWING EVALUATION RESULTS     ")
    print("=" * 45)
    print(f"Images Evaluated : {n}")
    print(f"Average PSNR     : {total_psnr / n:.3f} dB")
    print(f"Average SSIM     : {total_ssim / n:.4f}")
    print(f"RMSE (All) [LAB] : {np.mean(rmse_all_list):.3f}")
    if has_mask and len(rmse_shadow_list) > 0:
        print(f"RMSE (Shadow)    : {np.mean(rmse_shadow_list):.3f}")
        print(f"RMSE (Non-Shadow): {np.mean(rmse_non_shadow_list):.3f}")
    print("=" * 45)

if __name__ == "__main__":
    main()
