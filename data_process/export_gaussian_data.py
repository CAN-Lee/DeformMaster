import argparse
import os
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/data_process/data_config.csv")
parser.add_argument("--case-name", "--case_name", default=None,
                    help="Export only this case from the config.")
parser.add_argument("--base-path", default="./data/different_types")
parser.add_argument("--output-path", default="./data/gaussian_data")
parser.add_argument("--cams", default="0,1,2,3,4",
                    help="Comma-separated camera indices to export (default: 0,1,2,3,4)")
parser.add_argument("--controller-name", default="hand",
                    help="Controller label used in mask_info (e.g. 'hand' for "
                         "human demos, 'gripper' for robot demos)")
parser.add_argument("--controller-prompt", default="",
                    help="Text prompt to segment out the controller for Gaussian "
                         "export. Empty = auto: 'human' when controller is 'hand', "
                         "else the controller name itself.")
args = parser.parse_args()
cam_ids = [int(x) for x in args.cams.split(",") if x.strip() != ""]

base_path = args.base_path
output_path = args.output_path
CONTROLLER_NAME = args.controller_name
# Human demos label the controller "hand" but segment it with the prompt "human";
# robot demos use the same word (e.g. "gripper") for both.
CONTROLLER_PROMPT = args.controller_prompt or (
    "human" if CONTROLLER_NAME == "hand" else CONTROLLER_NAME
)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    """Copy a required input and fail clearly when it is missing."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Required input not found: {src}")
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)


def run_python(script, *script_args):
    """Run a pipeline helper without shell interpolation."""
    subprocess.run([sys.executable, script, *map(str, script_args)], check=True)


ensure_dir(output_path)

with open(args.config, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if args.case_name and case_name != args.case_name:
            continue

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        if os.path.exists(f"{output_path}/{case_name}/observation.ply"):
            print(f"Skipping {case_name} (observation.ply already exists)")
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            meta = json.load(f)

        # Create the directory for the case
        ensure_dir(f"{output_path}/{case_name}")
        for i in cam_ids:
            # Copy the original RGB image
            copy_file(f"{base_path}/{case_name}/color/{i}/0.png",
                      f"{output_path}/{case_name}/{i}.png")
            # Copy the original mask image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_ids = [int(key) for key, value in data.items()
                       if value != CONTROLLER_NAME]
            if not obj_ids:
                print(f"  [WARN] cam {i}: no object detected, using empty mask")
                # Create empty mask
                import cv2
                img = cv2.imread(f"{base_path}/{case_name}/color/{i}/0.png")
                if img is not None:
                    empty = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.imwrite(f"{output_path}/{case_name}/mask_{i}.png", empty)
            elif len(obj_ids) == 1:
                mask_path = f"{base_path}/{case_name}/mask/{i}/{obj_ids[0]}/0.png"
                copy_file(mask_path, f"{output_path}/{case_name}/mask_{i}.png")
            else:
                # Merge multiple object masks
                import cv2
                merged = None
                for oid in obj_ids:
                    m = cv2.imread(f"{base_path}/{case_name}/mask/{i}/{oid}/0.png",
                                   cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        merged = m if merged is None else np.maximum(merged, m)
                if merged is not None:
                    cv2.imwrite(f"{output_path}/{case_name}/mask_{i}.png", merged)

            # Prepare the high-resolution image (crop by object mask to avoid OOM on big frames)
            run_python("data_process/image_upscale.py",
                       "--img_path", f"{base_path}/{case_name}/color/{i}/0.png",
                       "--mask_path", f"{output_path}/{case_name}/mask_{i}.png",
                       "--output_path", f"{output_path}/{case_name}/{i}_high.png",
                       "--category", category)
            # Prepare the segmentation mask of the high-resolution image
            run_python("data_process/segment_util_image.py",
                       "--img_path", f"{output_path}/{case_name}/{i}_high.png",
                       "--TEXT_PROMPT", category,
                       "--output_path", f"{output_path}/{case_name}/mask_{i}_high.png")

            # Copy the original depth image
            copy_file(f"{base_path}/{case_name}/depth/{i}/0.npy",
                      f"{output_path}/{case_name}/{i}_depth.npy")

            # Prepare the controller mask for the low-resolution image and high-resolution image
            # (kept under the mask_human_* filename for downstream compatibility)
            run_python("data_process/segment_util_image.py",
                       "--img_path", f"{output_path}/{case_name}/{i}.png",
                       "--TEXT_PROMPT", CONTROLLER_PROMPT,
                       "--output_path", f"{output_path}/{case_name}/mask_human_{i}.png")
            run_python("data_process/segment_util_image.py",
                       "--img_path", f"{output_path}/{case_name}/{i}_high.png",
                       "--TEXT_PROMPT", CONTROLLER_PROMPT,
                       "--output_path", f"{output_path}/{case_name}/mask_human_{i}_high.png")

        # Prepare the intrinsic and extrinsic parameters
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        intrinsics = meta["intrinsics"]
        data = {}
        data["c2ws"] = c2ws
        data["intrinsics"] = intrinsics
        with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
            pickle.dump(data, f)

        # Prepare the shape initialization data
        if shape_prior.lower() == "true":
            copy_file(f"{base_path}/{case_name}/shape/matching/final_mesh.glb",
                      f"{output_path}/{case_name}/shape_prior.glb")
        # Save the original pcd data into the world coordinate system
        obs_points = []
        obs_colors = []
        pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
        processed_mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
        data = np.load(pcd_path)
        with open(processed_mask_path, "rb") as f:
            processed_masks = pickle.load(f)
        # pcd / processed_masks are stored densely (positional), in the same
        # order as the cams that were actually segmented. cam_ids may be sparse
        # (e.g. 0,2,4); use enumerate so j positions into the dense arrays.
        n_dense = len(data["points"])
        if len(cam_ids) != n_dense:
            print(f"  [WARN] cam_ids={cam_ids} has {len(cam_ids)} entries but "
                  f"pcd holds {n_dense}; assuming cam_ids order matches pcd order")
        for j, i in enumerate(cam_ids):
            if j >= n_dense:
                break
            points = data["points"][j]
            colors = data["colors"][j]
            mask = processed_masks[0][j]["object"]
            obs_points.append(points[mask])
            obs_colors.append(colors[mask])

        obs_points = np.vstack(obs_points)
        obs_colors = np.vstack(obs_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)
        o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)
