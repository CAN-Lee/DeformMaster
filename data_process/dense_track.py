# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument(
    "--cams",
    type=str,
    default="",
    help="Comma-separated camera ids (e.g. '0,2,4'). Empty = all available.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

if args.cams.strip():
    cams = [int(c) for c in args.cams.split(",") if c.strip()]
else:
    cams = sorted(int(os.path.basename(p)) for p in glob.glob(f"{base_path}/{case_name}/depth/*")
                  if os.path.basename(p).isdigit())
assert len(cams) > 0, f"no camera depth dirs under {base_path}/{case_name}/depth"
print(f"[dense_track] using cams={cams}")
device = "cuda"


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in cams:
        print(f"Processing {i}th camera")
        # Load the video
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        # Load the first-frame mask to get all query points from all masks
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # Revert x and y
        query_pixels = query_pixels[:, ::-1]
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        # Randomly select 5000 query points
        query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

        # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        # pred_tracks, pred_visibility = cotracker(video, queries=query_pixels[None], backward_tracking=True)
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        # Save the tracking data into npz
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].cpu().numpy()
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
