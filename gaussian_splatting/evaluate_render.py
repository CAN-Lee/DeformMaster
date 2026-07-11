import os
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
import json
from tqdm import tqdm
import torch
# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_MODEL = None


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def get_lpips_model():
    global LPIPS_MODEL
    if LPIPS_MODEL is None:
        LPIPS_MODEL = LPIPS("alex", "0.1").to(DEVICE)
        LPIPS_MODEL.eval()
    return LPIPS_MODEL


if __name__ == "__main__":
    import argparse

    def resolve_render_dir(prediction_dir, render_dir_arg):
        if render_dir_arg:
            return render_dir_arg

        if prediction_dir:
            candidates = [
                os.path.join(prediction_dir, "gaussian_output_dynamic_mpm"),
                os.path.join(prediction_dir, "gaussian_output_dynamic"),
            ]
            for candidate in candidates:
                if os.path.isdir(candidate):
                    return candidate

            if os.path.basename(os.path.normpath(prediction_dir)) == "gaussian_output_dynamic_mpm":
                return prediction_dir
            if os.path.basename(os.path.normpath(prediction_dir)) == "gaussian_output_dynamic":
                return prediction_dir
        
        if prediction_dir and "mpm" in prediction_dir:
            return './gaussian_output_dynamic_mpm'
        return './gaussian_output_dynamic'
    
    parser = argparse.ArgumentParser(description="Evaluate rendering quality for predictions")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Directory containing prediction results (optional, used to determine the render directory and output filename suffix)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output log file path (default: results/output_dynamic.txt or results/output_dynamic_mpm.txt if using an MPM prediction directory)"
    )
    parser.add_argument(
        "--render_dir",
        type=str,
        default=None,
        help="Explicit render output directory. If omitted, auto-resolve from prediction_dir."
    )
    args = parser.parse_args()
    
    render_path = './data/render_eval_data'
    human_mask_path = "./data/different_types_human_mask"
    root_data_dir = './data/gaussian_data'
    
    output_dir = resolve_render_dir(args.prediction_dir, args.render_dir)

    log_dir = './results'
    os.makedirs(log_dir, exist_ok=True)
    
    # Auto-generate output filename based on prediction_dir
    if args.output_file is None:
        if output_dir.endswith("gaussian_output_dynamic_mpm"):
            log_file_path = os.path.join(log_dir, 'output_dynamic_mpm.txt')
        else:
            log_file_path = os.path.join(log_dir, 'output_dynamic.txt')
    else:
        log_file_path = args.output_file

    print(f"Using render output directory: {output_dir}")
    print(f"Using device: {DEVICE}")
    lpips_model = get_lpips_model()

    with open(log_file_path, 'w') as log_file:

        scene_name = sorted(os.listdir(render_path))

        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

        scene_metrics = {}

        for scene in scene_name:

            scene_dir = os.path.join(root_data_dir, scene)
            output_scene_dir = os.path.join(output_dir, scene)
            render_path_dir = os.path.join(render_path, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)

            # Check if GT data exists
            if not os.path.exists(render_path_dir):
                print(f"Skipping {scene} (GT data not found in {render_path_dir})")
                continue

            # Load frame split info
            with open(f"{render_path_dir}/split.json", 'r') as f:
                info = json.load(f)
            frame_len = info['frame_len']
            train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))   # +1 if ignoring the first frame
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            # Check if render output exists for this scene
            if not os.path.exists(output_scene_dir):
                print(f"Skipping {scene} (render output directory not found: {output_scene_dir})")
                continue

            # Check if at least one render file exists (check first frame of first view)
            first_render_file = os.path.join(output_scene_dir, "0", f"{train_f_idx_range[0]:05d}.png")
            if not os.path.exists(first_render_file):
                print(f"Skipping {scene} (render files not found in {output_scene_dir}, expected file: {first_render_file})")
                continue

            print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
            print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

            psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            # for view_idx in range(3):
            for view_idx in range(1):   # only consider the first view

                for frame_idx in train_f_idx_range:
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_train.append(psnr(render_tensor, gt_tensor).item())
                    ssims_train.append(ssim(render_tensor, gt_tensor).item())
                    with torch.no_grad():
                        lpipss_train.append(lpips_model(render_tensor, gt_tensor).item())
                    ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

                for frame_idx in test_f_idx_range:
                        
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_test.append(psnr(render_tensor, gt_tensor).item())
                    ssims_test.append(ssim(render_tensor, gt_tensor).item())
                    with torch.no_grad():
                        lpipss_test.append(lpips_model(render_tensor, gt_tensor).item())
                    ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

            scene_metrics[scene] = {
                'psnr_train': np.mean(psnrs_train),
                'ssim_train': np.mean(ssims_train),
                'lpips_train': np.mean(lpipss_train),
                'iou_train': np.mean(ious_train),
                'psnr_test': np.mean(psnrs_test),
                'ssim_test': np.mean(ssims_test),
                'lpips_test': np.mean(lpipss_test),
                'iou_test': np.mean(ious_test)
            }

            all_psnrs_train.extend(psnrs_train)
            all_ssims_train.extend(ssims_train)
            all_lpipss_train.extend(lpipss_train)
            all_ious_train.extend(ious_train)

            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f'===== Scene: {scene} =====')
            print(f'\t PSNR (train): {np.mean(psnrs_train):.4f}')
            print(f'\t SSIM (train): {np.mean(ssims_train):.4f}')
            print(f'\t LPIPS (train): {np.mean(lpipss_train):.4f}')
            print(f'\t IoU (train): {np.mean(ious_train):.4f}')

            print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
            print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
            print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
            print(f'\t IoU (test): {np.mean(ious_test):.4f}')

        print('===== Overall Results Across All Scenes =====')
        print(f'\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}')
        print(f'\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}')
        print(f'\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}')
        print(f'\t Overall IoU (train): {np.mean(all_ious_train):.4f}')

        print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
        print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
        print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
        print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')

        overall_psnr_train = np.mean(all_psnrs_train)
        overall_ssim_train = np.mean(all_ssims_train)
        overall_lpips_train = np.mean(all_lpipss_train)
        overall_iou_train = np.mean(all_ious_train)
        
        overall_psnr_test = np.mean(all_psnrs_test)
        overall_ssim_test = np.mean(all_ssims_test)
        overall_lpips_test = np.mean(all_lpipss_test)
        overall_iou_test = np.mean(all_ious_test)

        # Write overall metrics to log file
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES\n")
        log_file.write("=" * 80 + "\n\n")
        
        log_file.write(f"Overall PSNR (train): {overall_psnr_train:.6f}\n")
        log_file.write(f"Overall SSIM (train): {overall_ssim_train:.6f}\n")
        log_file.write(f"Overall LPIPS (train): {overall_lpips_train:.6f}\n")
        log_file.write(f"Overall IoU (train): {overall_iou_train:.6f}\n\n")
        
        log_file.write(f"Overall PSNR (test): {overall_psnr_test:.6f}\n")
        log_file.write(f"Overall SSIM (test): {overall_ssim_test:.6f}\n")
        log_file.write(f"Overall LPIPS (test): {overall_lpips_test:.6f}\n")
        log_file.write(f"Overall IoU (test): {overall_iou_test:.6f}\n\n")
        
        # Create a compact table of all scene metrics
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Header
        log_file.write(f"{'Scene':<50} | {'PSNR-train':<12} | {'SSIM-train':<12} | {'LPIPS-train':<14} | {'IoU-train':<12} | ")
        log_file.write(f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n")
        log_file.write("-" * 160 + "\n")
        
        # Scene rows - only iterate over scenes that were actually processed
        for scene in scene_metrics.keys():
            metrics = scene_metrics[scene]
            log_file.write(f"{scene[:50]:<50} | ")
            log_file.write(f"{metrics['psnr_train']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_train']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_train']:<14.6f} | ")
            log_file.write(f"{metrics['iou_train']:<12.6f} | ")
            
            log_file.write(f"{metrics['psnr_test']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_test']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_test']:<14.6f} | ")
            log_file.write(f"{metrics['iou_test']:<12.6f}\n")
        
        # Overall row
        log_file.write("-" * 160 + "\n")
        log_file.write(f"{'OVERALL':<50} | ")
        log_file.write(f"{overall_psnr_train:<12.6f} | ")
        log_file.write(f"{overall_ssim_train:<12.6f} | ")
        log_file.write(f"{overall_lpips_train:<14.6f} | ")
        log_file.write(f"{overall_iou_train:<12.6f} | ")
        
        log_file.write(f"{overall_psnr_test:<12.6f} | ")
        log_file.write(f"{overall_ssim_test:<12.6f} | ")
        log_file.write(f"{overall_lpips_test:<14.6f} | ")
        log_file.write(f"{overall_iou_test:<12.6f}\n")
        
        print(f"\nMetrics have been saved to: {log_file_path}")
