from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.graphics_utils import focal2fov


@dataclass
class BridgeFrameSample:
    frame_idx: int
    cam_idx: int
    camera: Camera
    controller_mask: torch.Tensor


def _load_mask(path: str, size: Tuple[int, int]) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros(size, dtype=np.float32)
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    return (mask > 0).astype(np.float32)


class BridgeSequenceDataset(Dataset):
    def __init__(self, cfg, case_name: str, split: str = "train"):
        self.cfg = cfg
        self.case_name = case_name
        self.split = split
        self.case_dir = os.path.join(cfg.data.root, case_name)

        with open(os.path.join(self.case_dir, "split.json"), "r") as f:
            split_info = json.load(f)
        start, end = split_info[split]
        self.frame_indices = list(range(start, end))

        with open(os.path.join(self.case_dir, "calibrate.pkl"), "rb") as f:
            self.c2ws = pickle.load(f)
        with open(os.path.join(self.case_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        self.intrinsics = [np.array(intr, dtype=np.float32) for intr in metadata["intrinsics"]]
        self.width, self.height = metadata["WH"]

        # Allow multiple non-"hand" mask ids per cam (e.g. GroundedSAM2 may
        # split one physical object into several pieces). All are unioned
        # into a single object mask in __getitem__.
        self.object_ids: Dict[int, List[int]] = {}
        self.controller_ids: Dict[int, List[int]] = {}
        self.num_cams = len(self.intrinsics)
        for cam_idx in range(self.num_cams):
            with open(os.path.join(self.case_dir, "mask", f"mask_info_{cam_idx}.json"), "r") as f:
                mask_info = json.load(f)
            obj_ids = [int(k) for k, v in mask_info.items() if v != "hand"]
            if len(obj_ids) == 0:
                raise ValueError(
                    f"{case_name} cam {cam_idx}: no non-hand mask id, got {mask_info}"
                )
            self.object_ids[cam_idx] = obj_ids
            self.controller_ids[cam_idx] = [int(k) for k, v in mask_info.items() if v == "hand"]

        self.samples = [(frame_idx, cam_idx) for frame_idx in self.frame_indices for cam_idx in range(self.num_cams)]
        self.sample_to_index = {sample: idx for idx, sample in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> BridgeFrameSample:
        frame_idx, cam_idx = self.samples[index]
        rgb_path = os.path.join(self.case_dir, "color", str(cam_idx), f"{frame_idx}.png")
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb)

        object_mask = np.zeros((rgb_np.shape[0], rgb_np.shape[1]), dtype=np.float32)
        for obj_id in self.object_ids[cam_idx]:
            object_mask = np.maximum(
                object_mask,
                _load_mask(
                    os.path.join(self.case_dir, "mask", str(cam_idx), str(obj_id), f"{frame_idx}.png"),
                    size=(rgb_np.shape[0], rgb_np.shape[1]),
                ),
            )

        controller_mask = np.zeros_like(object_mask)
        for ctrl_id in self.controller_ids[cam_idx]:
            controller_mask = np.maximum(
                controller_mask,
                _load_mask(
                    os.path.join(self.case_dir, "mask", str(cam_idx), str(ctrl_id), f"{frame_idx}.png"),
                    size=(rgb_np.shape[0], rgb_np.shape[1]),
                ),
            )

        rgba = np.concatenate([rgb_np, (object_mask * 255.0).astype(np.uint8)[..., None]], axis=-1)
        image_rgba = Image.fromarray(rgba)

        c2w = np.asarray(self.c2ws[cam_idx], dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = self.intrinsics[cam_idx].copy()
        fov_y = focal2fov(K[1, 1], rgb_np.shape[0])
        fov_x = focal2fov(K[0, 0], rgb_np.shape[1])

        occ_mask = torch.from_numpy(controller_mask).float()
        camera = Camera(
            resolution=(rgb_np.shape[1], rgb_np.shape[0]),
            colmap_id=cam_idx,
            R=R,
            T=T,
            FoVx=fov_x,
            FoVy=fov_y,
            depth_params=None,
            image=image_rgba,
            invdepthmap=None,
            image_name=f"cam{cam_idx}_{frame_idx}",
            uid=index,
            train_test_exp=False,
            is_test_dataset=self.split != "train",
            is_test_view=self.split != "train",
            K=K,
            normal=None,
            depth=None,
            occ_mask=occ_mask,
        )
        return BridgeFrameSample(
            frame_idx=frame_idx,
            cam_idx=cam_idx,
            camera=camera,
            controller_mask=occ_mask,
        )

    def get_sample(self, frame_idx: int, cam_idx: int) -> BridgeFrameSample:
        return self[self.sample_to_index[(frame_idx, cam_idx)]]
