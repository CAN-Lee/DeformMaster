"""Gradio playground for interactive MPM + ResidualPGND + dynamic GS rendering."""

import collections
import os

# Use OSMesa for off-screen OpenGL rendering on headless servers.
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/tmp/gradio")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import json
import pickle
import queue
import random
import socket
import threading
import time
import warnings
import argparse
from argparse import ArgumentParser

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.dynamic_utils import (
    calc_weights_vals_from_indices,
    get_topk_indices,
    interpolate_motions_speedup,
    knn_weights_sparse,
)
from scripts_training_eval.appearance.gs_render import remove_gaussians_with_low_opacity
from deformmaster.engine.playground_engine import PlaygroundEngine
from deformmaster.model.diff_simulator.warp_solver.warp_utils import (
    torch2warp_float,
    torch2warp_vec3,
)
from deformmaster.utils.mpm_utils import youngs_poisson_to_lame


warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor", module="warp")

EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
VALID_KEYS = {"w", "a", "s", "d", "q", "e", "i", "j", "k", "l", "u", "o"}
DEFAULT_CONFIG_PATH = "configs/softbody.yaml"
KNOWN_CONFIG_PATHS = [
    "configs/softbody.yaml",
    "configs/cloth.yaml",
    "configs/rope.yaml",
]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_gs_view(w2c, intrinsic, height, width, device="cuda"):
    r_mat = np.transpose(w2c[:3, :3])
    t_vec = w2c[:3, 3]
    k_mat = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    fov_y = focal2fov(k_mat[1, 1], height)
    fov_x = focal2fov(k_mat[0, 0], width)
    return Camera(
        (width, height),
        colmap_id="0000",
        R=r_mat,
        T=t_vec,
        FoVx=fov_x,
        FoVy=fov_y,
        depth_params=None,
        image=None,
        invdepthmap=None,
        image_name="0000",
        uid="0000",
        data_device=device,
        train_test_exp=None,
        is_test_dataset=None,
        is_test_view=None,
        K=k_mat,
        normal=None,
        depth=None,
        occ_mask=None,
    )


def orbit_w2c(w2c, angle_deg, axis="y", pivot=None):
    """Return a new w2c matrix by orbiting the camera around *pivot*.

    The camera position (and orientation) is rotated by *angle_deg* degrees
    around *pivot* along the given axis while keeping the camera pointed at
    that pivot.  When *pivot* is None the orbit falls back to the world
    origin (legacy behaviour).
    """
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "y":
        R_orbit = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    elif axis == "x":
        R_orbit = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    else:
        R_orbit = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    if pivot is None:
        p = np.zeros(3, dtype=np.float64)
    else:
        p = np.asarray(pivot, dtype=np.float64).reshape(3)

    c2w = np.linalg.inv(w2c.astype(np.float64))
    c2w_new = np.eye(4, dtype=np.float64)
    c2w_new[:3, :3] = R_orbit @ c2w[:3, :3]
    c2w_new[:3, 3] = p + R_orbit @ (c2w[:3, 3] - p)
    return np.linalg.inv(c2w_new).astype(np.float32)


def build_status_text(pressed_keys, step_idx, recording, elapsed_sec, slot_states, fps):
    """Return an HTML status block to be displayed in a side panel
    (so it does not overlap the rendered scene and does not break click coordinates)."""
    held = " ".join(sorted(pressed_keys)).upper() if pressed_keys else "-"
    rec_html = (
        f"<span style='color:#d62020;font-weight:bold;'>REC {elapsed_sec:.1f}s</span>"
        if recording
        else "<span style='color:#1e9e1e;font-weight:bold;'>REC off</span>"
    )
    rows = [
        f"<b>Step:</b> {step_idx}",
        f"<b>FPS:</b> {fps:.1f}",
        f"<b>Ctrl 1:</b> W/A/S/D/Q/E",
        f"<b>Ctrl 2:</b> I/J/K/L/U/O",
        f"<b>Controllers:</b> {slot_states}",
        f"<b>Held:</b> {held}",
        rec_html,
    ]
    return (
        "<div style='font-family:monospace;font-size:13px;line-height:1.6;"
        "background:#ffffff;color:#000000;padding:14px;border-radius:6px;"
        "border:1px solid #cccccc;'>"
        "<div style='font-size:14px;font-weight:bold;color:#666;margin-bottom:6px;'>Live Status</div>"
        + "<br>".join(rows)
        + "</div>"
    )


def draw_status_overlay(frame, pressed_keys, step_idx, recording, elapsed_sec, slot_states, fps):
    """Backward-compat shim: returns the frame unchanged. Status info is now
    rendered as a separate Gradio HTML component (build_status_text)."""
    return frame


class GradioInteractiveMPMPlayground:
    def __init__(
        self,
        trainer,
        gs_path,
        intrinsic,
        w2c,
        width,
        height,
        overlay_image,
        n_ctrl_parts=1,
        inv_ctrl=False,
        output_dir="playground_recording/physics_flow",
        settle_iters=220,
        control_mode="mouse",
        rec_max_frames=0,
        rec_no_trail=True,
    ):
        self.trainer = trainer
        self.gs_path = gs_path
        self.n_ctrl_parts = n_ctrl_parts
        self.inv_ctrl = inv_ctrl
        self.output_dir = output_dir
        self.settle_iters = max(0, int(settle_iters))

        self.intrinsic = intrinsic
        self.w2c = w2c
        self.width = width
        self.height = height
        self.overlay_image = overlay_image
        self.control_mode = control_mode
        self.offset = self.trainer.auto_offset.detach().clone()
        self.reverse_z = bool(getattr(self.trainer.cfg.data, "reverse_z", False))
        # UI scale so controller markers/trails stay visually consistent across
        # different capture resolutions (e.g. mono 1080p vs multi-cam ~480p).
        # Reference height = 480 (historical multi-cam default) → scale 1.0.
        self._ui_scale = max(0.5, min(4.0, min(self.width, self.height) / 480.0))

        self.frame_queue = queue.Queue(maxsize=10)
        self.left_frame_queue = queue.Queue(maxsize=4)
        self.right_frame_queue = queue.Queue(maxsize=4)
        self.control_queue = queue.Queue()
        self.sim_lock = threading.Lock()
        self.running = False
        self.sim_thread = None
        self.last_frame = None
        self.last_left_frame = None
        self.last_right_frame = None
        self.pressed_keys = set()
        self.step_idx = 0
        # User-tunable substep count. Defaults to the cfg value; lowering
        # speeds up wall-clock per frame at the cost of physical time per
        # frame (motion plays slower). Damping is rescaled in
        # _step_physics so total per-frame energy loss stays comparable.
        self.steps_per_frame_override = int(trainer.cfg.mpm.steps_per_frame)

        self.recording = False
        self.rec_lock = threading.Lock()
        self.recorded_frames = []
        self.recorded_particles = []
        self.recorded_controllers = []
        self.record_count = 0
        self.rec_fps = 15.0
        self.rec_max_frames = int(rec_max_frames)
        self.rec_no_trail = bool(rec_no_trail)
        self._rec_last_time = 0.0
        self.slot_targets = [None for _ in range(self.n_ctrl_parts)]
        self.last_particles = None
        self._settled_pos = None
        self._init_pos = None
        self.hist_len = 2
        self.display_fps = 0.0
        self._last_loop_end = None
        self.material_controls = {
            "e_scale": 1.0,
            "nu_delta": 0.0,
            "fiber_scale": 1.0,
            "ctrl_stiffness_scale": 10.0,
            "ctrl_damping_scale": 1.0,
        }
        self._material_dirty = True

    def _to_original_frame(self, pos):
        if pos is None:
            return None
        # Invert the sim-frame pipeline (raw → reverse_z → @ R.T → + offset):
        #   sim → (- offset) → (@ R to undo mono_align rotation) → (reverse_z back).
        out = pos.detach().clone() - self.offset
        mono_R = getattr(self.trainer, '_mono_align_R', None)
        if mono_R is not None:
            mono_R = mono_R.to(device=out.device, dtype=out.dtype)
            out = out @ mono_R
        if self.reverse_z:
            out[..., 2] *= -1.0
        return out

    def _to_sim_frame(self, pos):
        if pos is None:
            return None
        out = pos.detach().clone()
        if self.reverse_z:
            out[..., 2] *= -1.0
        mono_R = getattr(self.trainer, '_mono_align_R', None)
        if mono_R is not None:
            mono_R = mono_R.to(device=out.device, dtype=out.dtype)
            out = out @ mono_R.T
        out = out + self.offset
        return out

    def _build_moe_wp(self):
        (
            w_patch,
            mu_patch,
            lam_patch,
            fk_patch,
            fdir_patch,
            _friction,
            yield_patch,
            e_patch,
            nu_patch,
            visc_patch,
            ctrl_stiffness_t,
            ctrl_damping_t,
        ) = self.trainer.get_current_phys_props()

        def gather_and_interp(patch_data):
            flat_idx = self.trainer.patch_idx.squeeze(0).view(-1)
            gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
            return torch.sum(self.trainer.interp_weights * gathered, dim=2).squeeze(0)

        p_weights = gather_and_interp(w_patch)
        p_e = gather_and_interp(e_patch.unsqueeze(-1)).squeeze()
        p_nu = gather_and_interp(nu_patch.unsqueeze(-1)).squeeze()
        bounds = self.trainer.cfg.mpm.get("material_bounds", {})
        nu_min, nu_max = bounds.get("nu", [0.0, 0.45])
        p_e = torch.clamp(p_e * float(self.material_controls["e_scale"]), min=1e-6)
        p_nu = torch.clamp(p_nu + float(self.material_controls["nu_delta"]), min=nu_min, max=nu_max)
        p_mu, p_lam = youngs_poisson_to_lame(p_e, p_nu)
        p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze() * float(self.material_controls["fiber_scale"])
        p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
        ctrl_stiffness_t = ctrl_stiffness_t * float(self.material_controls["ctrl_stiffness_scale"])
        ctrl_damping_t = ctrl_damping_t * float(self.material_controls["ctrl_damping_scale"])

        expert_order = ["nh", "co", "st", "fi"]
        active = getattr(self.trainer.cfg.mpm, "active_experts", expert_order)
        mask_active = [1 if e in active else 0 for e in expert_order]
        active_mask_wp = wp.array(mask_active, dtype=wp.int32, device=self.trainer.simulator.warp_device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moe_params_wp = {
                "weights": torch2warp_float(p_weights),
                "mu": torch2warp_float(p_mu),
                "lam": torch2warp_float(p_lam),
                "fk": torch2warp_float(p_fk),
                "fdir": torch2warp_vec3(p_fdir),
                "active_mask": active_mask_wp,
            }

        return moe_params_wp, ctrl_stiffness_t, ctrl_damping_t, yield_patch, visc_patch

    def _refresh_material_cache_if_needed(self):
        if not self._material_dirty:
            return
        self.moe_params_wp, self.ctrl_stiffness_t, self.ctrl_damping_t, _, _ = self._build_moe_wp()
        self._material_dirty = False

    def set_material_controls(self, e_scale, nu_delta, fiber_scale, ctrl_stiffness_scale, ctrl_damping_scale):
        with self.sim_lock:
            self.material_controls = {
                "e_scale": float(e_scale),
                "nu_delta": float(nu_delta),
                "fiber_scale": float(fiber_scale),
                "ctrl_stiffness_scale": float(ctrl_stiffness_scale),
                "ctrl_damping_scale": float(ctrl_damping_scale),
            }
            self._material_dirty = True
        return (
            "Material controls updated: "
            f"E x{float(e_scale):.2f}, nu {float(nu_delta):+.2f}, fiber x{float(fiber_scale):.2f}, "
            f"kp x{float(ctrl_stiffness_scale):.2f}, kd x{float(ctrl_damping_scale):.2f}"
        )

    def _reset_hist_from(self, x0):
        self.x_history = [x0.clone() for _ in range(self.hist_len)]
        self.v_history = [torch.zeros_like(x0) for _ in range(self.hist_len)]

    def _ensure_settled(self):
        if self._settled_pos is not None:
            return self._settled_pos

        assert self._init_pos is not None, "Initial positions must be prepared before settling."
        print(f"[interactive_mpm] Settling for {self.settle_iters} ticks with gravity only...")
        self.trainer.simulator.reset(self._init_pos.clone(), controller_pos=None)
        x0 = (self.trainer.simulator.x - self.trainer.simulator.shift).detach()
        self._reset_hist_from(x0)
        with torch.no_grad():
            for _ in range(self.settle_iters):
                self._step_physics(None, None)
        self._settled_pos = (self.trainer.simulator.x - self.trainer.simulator.shift).detach().clone()
        print("[interactive_mpm] Settling complete. Cached grounded rest pose.")
        return self._settled_pos

    def setup_simulation(self):
        trainer = self.trainer
        sim = trainer.simulator

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=trainer.device)
        view = build_gs_view(self.w2c, self.intrinsic, self.height, self.width, device=str(trainer.device))
        overlay = self.overlay_image.to(trainer.device)

        init_pos = (trainer.data["init_pos"].to(trainer.device) + trainer.auto_offset).detach()

        # Side-view cameras (small orbital offsets for left/right panels), pivoted on the object centre
        self.side_angle_deg = 30.0
        try:
            init_pivot = self._to_original_frame(init_pos).mean(dim=0).detach().cpu().numpy().astype(np.float64)
        except Exception:
            init_pivot = None
        self.left_w2c = orbit_w2c(self.w2c, self.side_angle_deg, axis="z", pivot=init_pivot)
        self.right_w2c = orbit_w2c(self.w2c, -self.side_angle_deg, axis="z", pivot=init_pivot)
        left_view = build_gs_view(self.left_w2c, self.intrinsic, self.height, self.width, device=str(trainer.device))
        right_view = build_gs_view(self.right_w2c, self.intrinsic, self.height, self.width, device=str(trainer.device))

        self._init_pos = init_pos.clone()

        sim.reset(init_pos.clone(), controller_pos=None)
        self.moe_params_wp, self.ctrl_stiffness_t, self.ctrl_damping_t, _, _ = self._build_moe_wp()

        self.hist_len = getattr(
            trainer.cfg.residual if hasattr(trainer.cfg, "residual") else None,
            "n_history",
            2,
        )
        x0 = (sim.x - sim.shift).detach()
        self._reset_hist_from(x0)
        settled_pos = self._ensure_settled().clone()
        sim.reset(settled_pos.clone(), controller_pos=None)
        x0 = (sim.x - sim.shift).detach()
        self._reset_hist_from(x0)

        inv = -1.0 if self.inv_ctrl else 1.0
        self.key_mappings = {
            "w": (0, np.array([0.005, 0.0, 0.0], dtype=np.float32) * inv),
            "s": (0, np.array([-0.005, 0.0, 0.0], dtype=np.float32) * inv),
            "a": (0, np.array([0.0, -0.005, 0.0], dtype=np.float32) * inv),
            "d": (0, np.array([0.0, 0.005, 0.0], dtype=np.float32) * inv),
            "q": (0, np.array([0.0, 0.0, -0.005], dtype=np.float32)),
            "e": (0, np.array([0.0, 0.0, 0.005], dtype=np.float32)),
            "i": (1, np.array([0.005, 0.0, 0.0], dtype=np.float32) * inv),
            "k": (1, np.array([-0.005, 0.0, 0.0], dtype=np.float32) * inv),
            "j": (1, np.array([0.0, -0.005, 0.0], dtype=np.float32) * inv),
            "l": (1, np.array([0.0, 0.005, 0.0], dtype=np.float32) * inv),
            "u": (1, np.array([0.0, 0.0, -0.005], dtype=np.float32)),
            "o": (1, np.array([0.0, 0.0, 0.005], dtype=np.float32)),
        }

        self.step_idx = 0
        self.display_fps = 0.0
        self._last_loop_end = None
        self.slot_targets = [None for _ in range(self.n_ctrl_parts)]
        # Per-controller world-frame trajectory history (for visualization overlay)
        self.controller_history = [collections.deque(maxlen=60) for _ in range(self.n_ctrl_parts)]
        self.last_particles = x0.clone()
        self.sim_state = {
            "gaussians": gaussians,
            "view": view,
            "background": background,
            "overlay": overlay,
            "current_target": None,
            "prev_target": None,
            "prev_x": None,
            "relations": None,
            "weights": None,
            "weights_indices": None,
            "width": self.width,
            "height": self.height,
            "intrinsic": self.intrinsic,
            "w2c": self.w2c,
            "use_white_background": True,
            "init_pos": settled_pos.clone(),
            "left_view": left_view,
            "right_view": right_view,
            "left_w2c": self.left_w2c,
            "right_w2c": self.right_w2c,
        }

        self._base_gs_xyz = gaussians.get_xyz.clone()
        self._base_gs_rot = gaussians.get_rotation.clone()
        # Cache pristine opacity ONCE (before any fracture mask edits)
        self._base_gs_opacity_raw = gaussians._opacity.detach().clone()
        settled_gs_xyz, settled_gs_rot = self._apply_gs_pose_from_particles(init_pos, settled_pos)
        self._init_gs_xyz = settled_gs_xyz.clone()
        self._init_gs_rot = settled_gs_rot.clone()
        self.sim_state["prev_x"] = x0.clone()
        # Reset fracture-detection baseline; will be re-established on first
        # _update_gs_from_particles call.
        self._init_bones_for_lbs = None

    def _get_target_change(self):
        if self.control_mode != "keyboard":
            if self.n_ctrl_parts == 1:
                return np.zeros(3, dtype=np.float32)
            return np.zeros((self.n_ctrl_parts, 3), dtype=np.float32)
        target_change = np.zeros((self.n_ctrl_parts, 3), dtype=np.float32)
        for key in self.pressed_keys:
            if key not in self.key_mappings:
                continue
            part_idx, delta = self.key_mappings[key]
            if part_idx < self.n_ctrl_parts:
                target_change[part_idx] += delta
        if self.n_ctrl_parts == 1:
            return target_change[0]
        return target_change

    def _packed_slot_targets(self):
        packed = []
        slot_to_packed = {}
        for slot_idx, target in enumerate(self.slot_targets):
            if target is None:
                continue
            slot_to_packed[slot_idx] = len(packed)
            packed.append(target)
        if not packed:
            return None, slot_to_packed
        return torch.stack(packed, dim=0), slot_to_packed

    def _apply_target_change(self, current_target, target_change, slot_to_packed):
        if current_target is None:
            return None
        if self.n_ctrl_parts == 1:
            target_change = np.asarray(target_change, dtype=np.float32)[None]
        for slot_idx in range(self.n_ctrl_parts):
            packed_idx = slot_to_packed.get(slot_idx)
            if packed_idx is None:
                continue
            delta = torch.tensor(target_change[slot_idx], dtype=torch.float32, device=current_target.device)
            current_target[packed_idx] += delta
            self.slot_targets[slot_idx] = current_target[packed_idx].clone()
        return current_target

    def _slot_summary_text(self):
        states = []
        for idx, target in enumerate(self.slot_targets):
            states.append(f"{idx + 1}:{'set' if target is not None else '-'}")
        return " ".join(states)

    def set_inv_ctrl(self, invert):
        with self.sim_lock:
            self.inv_ctrl = bool(invert)
            inv = -1.0 if self.inv_ctrl else 1.0
            self.key_mappings = {
                "w": (0, np.array([0.005, 0.0, 0.0], dtype=np.float32) * inv),
                "s": (0, np.array([-0.005, 0.0, 0.0], dtype=np.float32) * inv),
                "a": (0, np.array([0.0, -0.005, 0.0], dtype=np.float32) * inv),
                "d": (0, np.array([0.0, 0.005, 0.0], dtype=np.float32) * inv),
                "q": (0, np.array([0.0, 0.0, -0.005], dtype=np.float32)),
                "e": (0, np.array([0.0, 0.0, 0.005], dtype=np.float32)),
                "i": (1, np.array([0.005, 0.0, 0.0], dtype=np.float32) * inv),
                "k": (1, np.array([-0.005, 0.0, 0.0], dtype=np.float32) * inv),
                "j": (1, np.array([0.0, -0.005, 0.0], dtype=np.float32) * inv),
                "l": (1, np.array([0.0, 0.005, 0.0], dtype=np.float32) * inv),
                "u": (1, np.array([0.0, 0.0, -0.005], dtype=np.float32)),
                "o": (1, np.array([0.0, 0.0, 0.005], dtype=np.float32)),
            }
        state = "inverted" if self.inv_ctrl else "normal"
        return f"Control direction: {state}"

    def set_control_mode(self, mode):
        mode = (mode or "mouse").strip().lower()
        if mode not in {"mouse", "keyboard"}:
            return f"Unsupported mode: {mode}"
        with self.sim_lock:
            self.control_mode = mode
            self.pressed_keys.clear()
        return f"Control mode set to {mode}"

    def _controllers_for_save(self):
        saved = np.full((self.n_ctrl_parts, 3), np.nan, dtype=np.float32)
        for idx, target in enumerate(self.slot_targets):
            if target is None:
                continue
            saved[idx] = self._to_original_frame(target).detach().cpu().numpy().astype(np.float32)
        return saved

    def _project_particles(self, particles):
        pts = self._to_original_frame(particles).detach().cpu().numpy()
        r_mat = self.w2c[:3, :3]
        t_vec = self.w2c[:3, 3]
        cam = (r_mat @ pts.T).T + t_vec
        valid = cam[:, 2] > 0.01
        proj = (self.intrinsic @ cam[valid].T).T
        uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
        uv_valid = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
        uv[valid] = uv_valid
        in_view = (
            valid
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.height)
        )
        return uv, in_view

    def select_controller_from_click(self, x_pix, y_pix, slot_idx):
        with self.sim_lock:
            if self.control_mode == "keyboard":
                return "Mouse click is disabled in keyboard mode. Switch to Mouse mode to pick controllers."
            if not self.running or self.last_particles is None:
                return "Start simulation first, then click the image."

            slot_idx = int(slot_idx)
            if slot_idx < 0 or slot_idx >= self.n_ctrl_parts:
                return f"Invalid slot index: {slot_idx + 1}"

            # In mouse mode, the click is ONLY for the initial pick. Once a slot
            # is bound, subsequent clicks on that slot are rejected so they don't
            # accidentally jump the controller during simulation. To re-pick,
            # press Reset.
            if self.slot_targets[slot_idx] is not None:
                return (
                    f"Controller {slot_idx + 1} already bound. "
                    "Press Reset to re-pick, or pick the other controller via the radio selector."
                )

            particles = self.last_particles
            uv, valid = self._project_particles(particles)
            if not np.any(valid):
                return "No visible particles available for selection."

            diff = uv[valid] - np.array([[x_pix, y_pix]], dtype=np.float32)
            local_idx = int(np.argmin(np.sum(diff * diff, axis=1)))
            particle_idx = np.flatnonzero(valid)[local_idx]
            particles_orig = self._to_original_frame(particles)
            selected_pos_orig = particles_orig[particle_idx].detach().clone()
            selected_pos = self._to_sim_frame(selected_pos_orig)

            self.slot_targets[slot_idx] = selected_pos
            packed_targets, _ = self._packed_slot_targets()
            current_x = particles.detach().clone()
            self.trainer.simulator.reset(current_x, controller_pos=packed_targets)

            x0 = (self.trainer.simulator.x - self.trainer.simulator.shift).detach()
            self._reset_hist_from(x0)
            self.last_particles = x0.clone()
            self.sim_state["current_target"] = packed_targets.clone() if packed_targets is not None else None
            self.sim_state["prev_target"] = packed_targets.clone() if packed_targets is not None else None
            self.sim_state["prev_x"] = None
            self.sim_state["relations"] = None
            self.sim_state["weights"] = None
            self.sim_state["weights_indices"] = None

            # Render a fresh frame immediately so the controller marker shows up
            # without waiting for the next physics iteration (which can be slow
            # if GS rendering is the bottleneck).
            try:
                fresh = self._render_frame(self.last_particles)
                if fresh is not None:
                    try:
                        self.frame_queue.put_nowait(fresh)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(fresh)
                        except queue.Empty:
                            pass
                    self.last_frame = fresh
            except Exception as e:
                print(f"[WARN] Failed to refresh frame after pick: {e}")

            return (
                f"Controller {slot_idx + 1} attached at "
                f"({selected_pos_orig[0].item():.3f}, {selected_pos_orig[1].item():.3f}, {selected_pos_orig[2].item():.3f})"
            )

    def _step_physics(self, prev_target, current_target):
        trainer = self.trainer
        sim = trainer.simulator

        dt = trainer.cfg.mpm.dt
        cfg_steps = int(trainer.cfg.mpm.steps_per_frame)
        steps_per_frame = max(1, int(self.steps_per_frame_override))
        # Rescale per-frame body damping so total per-frame energy loss
        # stays comparable when the user shrinks the substep count.
        damping_scale = cfg_steps / float(steps_per_frame)
        frame_dt = dt * steps_per_frame
        x_start_frame = (sim.x - sim.shift).detach().unsqueeze(0)

        warmup_frames = getattr(trainer.cfg.mpm, "controller_warmup_frames", 3)
        warmup_factor = min(1.0, (self.step_idx + 1) / (warmup_frames + 1e-6))
        current_stiffness = float(self.ctrl_stiffness_t.item()) * warmup_factor
        current_damping = float(self.ctrl_damping_t.item()) * warmup_factor * damping_scale

        with torch.no_grad():
            if current_target is None:
                v_ctrl_t = None
            elif prev_target is None:
                v_ctrl_t = torch.zeros_like(current_target)
            else:
                v_ctrl_t = (current_target - prev_target) / max(frame_dt, 1e-9)

            for s in range(steps_per_frame):
                if current_target is None:
                    curr_target_pos = None
                elif prev_target is None:
                    curr_target_pos = current_target
                else:
                    alpha = (s + 1) / steps_per_frame
                    curr_target_pos = prev_target + alpha * (current_target - prev_target)
                sim.step(
                    self.moe_params_wp,
                    controller_pos=curr_target_pos,
                    controller_vel=v_ctrl_t,
                    residual_v=None,
                    stiffness_override=current_stiffness,
                    damping_override=current_damping,
                )

            if trainer.residual_net is not None:
                trainer.residual_net.eval()
                x_his_tensor = torch.stack(self.x_history, dim=1).unsqueeze(0)
                v_his_tensor = torch.stack(self.v_history, dim=1).unsqueeze(0)
                curr_x_mpm = (sim.x - sim.shift).unsqueeze(0)
                curr_v_mpm = sim.v.unsqueeze(0)
                delta_v = trainer.residual_net(
                    curr_x_mpm,
                    curr_v_mpm,
                    x_start_frame,
                    x_his_tensor,
                    v_his_tensor,
                ).squeeze(0)
                sim.v = sim.v + delta_v
                sim.x = sim.x + delta_v * frame_dt

            sim.x = sim.x.detach()
            sim.v = sim.v.detach()
            sim.F = sim.F.detach()
            sim.C = sim.C.detach()

            x = (sim.x - sim.shift).detach()
            self.x_history.pop(0)
            self.x_history.append(x.clone())
            self.v_history.pop(0)
            self.v_history.append(sim.v.detach().clone())

        return x

    def _render_frame(self, x):
        state = self.sim_state
        x_render = self._to_original_frame(x)
        frame = state["overlay"].clone()
        results = render_gaussian(state["view"], state["gaussians"], None, state["background"])
        rendering = results["render"]
        image = rendering.permute(1, 2, 0).detach().clamp(0, 1)

        if state["use_white_background"]:
            image_mask = torch.logical_and((image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255)
        else:
            image_mask = torch.logical_and((image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255)
        image[..., 3].masked_fill_(~image_mask, 0.0)

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.cpu().numpy()
        image_mask = image_mask.cpu().numpy()
        frame = frame.astype(np.uint8)

        for light_point, factor in [([0, 0, -3], 0.95), ([1, 0.5, -2], 0.97), ([-3, -0.5, -5], 0.98)]:
            shadow = self._get_simple_shadow(
                x_render,
                state["intrinsic"],
                state["w2c"],
                state["width"],
                state["height"],
                image_mask,
                light_point=light_point,
            )
            frame[shadow] = (frame[shadow] * factor).astype(np.uint8)
        # Controller markers / trail are drawn by the sim loop so the
        # recorded video can use a different overlay (e.g. no trail) than
        # the live display.
        return frame

    def _render_side_frame(self, x, side="left"):
        """Render a simplified side-view frame (no shadows / controller markers)."""
        state = self.sim_state
        view = state["left_view"] if side == "left" else state["right_view"]
        results = render_gaussian(view, state["gaussians"], None, state["background"])
        rendering = results["render"]
        image = rendering.permute(1, 2, 0).detach().clamp(0, 1)

        # White background for novel views (no overlay image)
        frame = torch.full((state["height"], state["width"], 3), 255.0,
                           dtype=torch.float32, device=image.device)

        if state["use_white_background"]:
            image_mask = torch.logical_and((image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255)
        else:
            image_mask = torch.logical_and((image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255)
        image[..., 3].masked_fill_(~image_mask, 0.0)

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.cpu().numpy().astype(np.uint8)
        return frame

    def _project_world_to_pixel(self, world_xyz):
        """Project a single 3D world-frame point to pixel (x, y) or None if not visible."""
        r_mat = self.w2c[:3, :3]
        t_vec = self.w2c[:3, 3]
        cam = r_mat @ world_xyz + t_vec
        if cam[2] <= 0.01:
            return None
        uv = self.intrinsic @ cam
        x_pix = int(round(float(uv[0] / uv[2])))
        y_pix = int(round(float(uv[1] / uv[2])))
        if x_pix < 0 or x_pix >= self.width or y_pix < 0 or y_pix >= self.height:
            return None
        return (x_pix, y_pix)

    def _update_controller_history(self):
        # Record current targets into per-controller world-frame history.
        # Split out so drawing markers multiple times per step (e.g. once for
        # display with trail, once for recording without trail) doesn't
        # double-grow the history buffer.
        for slot_idx, target in enumerate(self.slot_targets):
            if target is None:
                continue
            world_xyz = self._to_original_frame(target).detach().cpu().numpy().astype(np.float32)
            self.controller_history[slot_idx].append(world_xyz)

    def _draw_controller_markers(self, frame, draw_trail=True):
        # `draw_trail` controls all the per-step overlays the saved recording
        # may want to suppress: history polyline, velocity arrow + speed text,
        # and the C1/C2 label. Only the position circle is always drawn.
        if all(len(h) == 0 for h in self.controller_history):
            return frame

        frame_out = frame.copy()
        slot_colors = [(60, 220, 255), (255, 180, 60)]  # cyan, orange

        # Resolution-adaptive sizing (scale 1.0 at 480p reference height).
        s = self._ui_scale
        r_outer = max(1, int(round(9 * s)))
        r_inner = max(1, int(round(3 * s)))
        t_outer = max(1, int(round(2 * s)))
        t_trail_thin = max(1, int(round(1 * s)))
        t_trail_thick = max(1, int(round(2 * s)))
        t_arrow = max(1, int(round(2 * s)))
        arrow_max = 80.0 * s
        arrow_scale = 8.0  # per-frame magnification, resolution-independent
        font_label = 0.55 * s
        font_speed = 0.45 * s
        t_font_label = max(1, int(round(2 * s)))
        t_font_speed = max(1, int(round(1 * s)))
        lbl_dx = int(round(10 * s))
        lbl_dy = int(round(10 * s))
        speed_dx = int(round(4 * s))

        for slot_idx in range(self.n_ctrl_parts):
            history = list(self.controller_history[slot_idx])
            if not history:
                continue
            color = slot_colors[slot_idx % len(slot_colors)]

            # 1) Project all history points to pixel space
            projected = [self._project_world_to_pixel(p) for p in history]
            n = len(projected)

            # 2) Trajectory polyline (older = dimmer / thinner)
            if draw_trail:
                for i in range(1, n):
                    p_prev = projected[i - 1]
                    p_curr = projected[i]
                    if p_prev is None or p_curr is None:
                        continue
                    age_factor = i / n
                    alpha = 0.2 + 0.8 * age_factor
                    thickness = t_trail_thin if age_factor < 0.5 else t_trail_thick
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame_out, p_prev, p_curr, trail_color, thickness, cv2.LINE_AA)

            # 3) Current position marker
            current_pix = projected[-1]
            if current_pix is None:
                continue
            x_pix, y_pix = current_pix
            cv2.circle(frame_out, (x_pix, y_pix), r_outer, color, t_outer, cv2.LINE_AA)
            cv2.circle(frame_out, (x_pix, y_pix), r_inner, color, -1, cv2.LINE_AA)

            # 4) Velocity arrow (direction + magnitude) from last 2 history points
            if draw_trail and n >= 2 and projected[-2] is not None:
                prev_x, prev_y = projected[-2]
                dx = x_pix - prev_x
                dy = y_pix - prev_y
                mag_pix = (dx * dx + dy * dy) ** 0.5
                if mag_pix > 0.5:
                    arrow_len = min(mag_pix * arrow_scale, arrow_max)
                    nx = dx / mag_pix
                    ny = dy / mag_pix
                    tip = (
                        int(round(x_pix + nx * arrow_len)),
                        int(round(y_pix + ny * arrow_len)),
                    )
                    arrow_color = (0, 255, 0)  # green (RGB), contrasts with cyan/orange markers
                    cv2.arrowedLine(
                        frame_out, (x_pix, y_pix), tip, arrow_color, t_arrow, cv2.LINE_AA, tipLength=0.3
                    )
                    history_arr = np.asarray(history[-2:])
                    world_speed = float(np.linalg.norm(history_arr[1] - history_arr[0]))
                    cv2.putText(
                        frame_out,
                        f"{world_speed*1000:.1f}mm/f",
                        (tip[0] + speed_dx, tip[1] + speed_dx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_speed,
                        arrow_color,
                        t_font_speed,
                        cv2.LINE_AA,
                    )

            # 5) Controller label
            if draw_trail:
                cv2.putText(
                    frame_out,
                    f"C{slot_idx + 1}",
                    (x_pix + lbl_dx, y_pix - lbl_dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_label,
                    color,
                    t_font_label,
                    cv2.LINE_AA,
                )

        return frame_out

    @staticmethod
    def _get_simple_shadow(x, intrinsic, w2c, width, height, image_mask, light_point):
        light = np.asarray(light_point, dtype=np.float32)
        pts = x.detach().cpu().numpy()
        light_to_pt = pts - light[None]
        eps = 1e-6
        scale = np.where(np.abs(light_to_pt[:, 2]) > eps, -light[2] / (light_to_pt[:, 2] + eps), 0.0)
        proj = light[None] + light_to_pt * scale[:, None]
        homog = np.concatenate([proj, np.ones((proj.shape[0], 1), dtype=np.float32)], axis=1)
        proj_mat = intrinsic @ w2c[:3, :]
        uvw = (proj_mat @ homog.T).T
        valid = uvw[:, 2] > 1e-6
        uv = np.zeros((proj.shape[0], 2), dtype=np.int32)
        uv[valid] = np.round(uvw[valid, :2] / uvw[valid, 2:3]).astype(np.int32)
        mask = np.zeros((height, width), dtype=bool)
        in_view = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        mask[uv[in_view, 1], uv[in_view, 0]] = True
        mask &= ~image_mask
        return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)

    def _apply_gs_pose_from_particles(self, source_particles, target_particles):
        state = self.sim_state
        with torch.no_grad():
            source_particles_orig = self._to_original_frame(source_particles)
            target_particles_orig = self._to_original_frame(target_particles)
            relations = get_topk_indices(source_particles_orig, K=16)
            weights, weights_indices = knn_weights_sparse(source_particles_orig, self._base_gs_xyz, K=16)
            weights = calc_weights_vals_from_indices(source_particles_orig, self._base_gs_xyz, weights_indices)
            current_pos, current_rot, _ = interpolate_motions_speedup(
                bones=source_particles_orig,
                motions=target_particles_orig - source_particles_orig,
                relations=relations,
                weights=weights,
                weights_indices=weights_indices,
                xyz=self._base_gs_xyz,
                quat=self._base_gs_rot,
            )
            state["gaussians"]._xyz = current_pos
            state["gaussians"]._rotation = current_rot
            return current_pos.clone(), current_rot.clone()

    def _update_gs_from_particles(self, prev_x, x):
        state = self.sim_state
        with torch.no_grad():
            prev_x_orig = self._to_original_frame(prev_x)
            x_orig = self._to_original_frame(x)

            if state["relations"] is None:
                state["relations"] = get_topk_indices(prev_x_orig, K=16)

            if state["weights"] is None:
                state["weights"], state["weights_indices"] = knn_weights_sparse(
                    prev_x_orig, state["gaussians"].get_xyz, K=16
                )

            state["weights"] = calc_weights_vals_from_indices(
                prev_x_orig, state["gaussians"].get_xyz, state["weights_indices"]
            )

            current_pos, current_rot, _ = interpolate_motions_speedup(
                bones=prev_x_orig,
                motions=x_orig - prev_x_orig,
                relations=state["relations"],
                weights=state["weights"],
                weights_indices=state["weights_indices"],
                xyz=state["gaussians"].get_xyz,
                quat=state["gaussians"].get_rotation,
            )
            state["gaussians"]._xyz = current_pos
            state["gaussians"]._rotation = current_rot

            # [Fracture cleanup] Hide Gaussians whose bones span a torn region.
            # Compare current bone-cluster spread vs initial spread; if it has
            # grown above STRETCH_THRESHOLD, the Gaussian is bridging a fracture
            # — set its raw opacity very negative so sigmoid(opacity) ≈ 0.
            self._apply_fracture_opacity_mask(x_orig)

    def _apply_fracture_opacity_mask(self, current_bones, stretch_threshold=3.0):
        state = self.sim_state
        gaussians = state["gaussians"]
        weights_indices = state.get("weights_indices")
        if weights_indices is None:
            return
        # Cache initial bone positions on first call (after init or reset).
        # Note: _base_gs_opacity_raw is cached ONCE in setup_simulation, never here.
        if not hasattr(self, "_init_bones_for_lbs") or self._init_bones_for_lbs is None:
            self._init_bones_for_lbs = current_bones.detach().clone()
            init_bones_per_gs = self._init_bones_for_lbs[weights_indices]  # (N_gs, K, 3)
            self._init_spread = init_bones_per_gs.std(dim=1).norm(dim=-1) + 1e-8  # (N_gs,)
            return  # nothing to mask on first call

        cur_bones_per_gs = current_bones[weights_indices]  # (N_gs, K, 3)
        cur_spread = cur_bones_per_gs.std(dim=1).norm(dim=-1)
        torn_mask = cur_spread > (stretch_threshold * self._init_spread)
        if torn_mask.any():
            new_opacity = self._base_gs_opacity_raw.clone()
            new_opacity[torn_mask] = -100.0  # sigmoid(-100) ~ 0
            gaussians._opacity = new_opacity
        else:
            gaussians._opacity = self._base_gs_opacity_raw.clone()

    def simulation_loop(self):
        state = self.sim_state
        current_target = state["current_target"]
        prev_target = state["prev_target"]
        prev_x = state["prev_x"]

        while self.running:
            try:
                while not self.control_queue.empty():
                    cmd = self.control_queue.get_nowait()
                    if cmd == "stop":
                        self.running = False
                        break
                    if cmd == "reset":
                        self.trainer.simulator.reset(state["init_pos"].clone(), controller_pos=None)
                        state["gaussians"]._xyz = self._init_gs_xyz.clone()
                        state["gaussians"]._rotation = self._init_gs_rot.clone()
                        # Restore pristine opacity so previously hidden parts reappear
                        if hasattr(self, "_base_gs_opacity_raw") and self._base_gs_opacity_raw is not None:
                            state["gaussians"]._opacity = self._base_gs_opacity_raw.clone()
                        self.slot_targets = [None for _ in range(self.n_ctrl_parts)]
                        for h in self.controller_history:
                            h.clear()
                        # Clear cached fracture-detection baseline so it
                        # re-establishes from the next frame.
                        self._init_bones_for_lbs = None
                        self.drag_active = False
                        self.drag_slot_idx = None
                        self.drag_depth = None
                        current_target = None
                        prev_target = None
                        prev_x = None
                        self.step_idx = 0
                        self.display_fps = 0.0
                        self._last_loop_end = None
                        state["relations"] = None
                        state["weights"] = None
                        state["weights_indices"] = None
                        x0 = (self.trainer.simulator.x - self.trainer.simulator.shift).detach()
                        self._reset_hist_from(x0)
                        self.last_particles = x0.clone()
                        state["prev_x"] = x0.clone()
                        self.pressed_keys.clear()
                        continue
                    if cmd.startswith("key:"):
                        key = cmd.split(":")[1]
                        if key in self.key_mappings:
                            self.pressed_keys.add(key)
                    if cmd.startswith("release:"):
                        key = cmd.split(":")[1]
                        self.pressed_keys.discard(key)

                if not self.running:
                    break

                with self.sim_lock:
                    self._refresh_material_cache_if_needed()
                    current_target, slot_to_packed = self._packed_slot_targets()
                    prev_target = None if current_target is None else current_target.clone()
                    target_change = self._get_target_change()
                    current_target = self._apply_target_change(current_target, target_change, slot_to_packed)
                    x = self._step_physics(prev_target, current_target)

                    if prev_x is not None:
                        self._update_gs_from_particles(prev_x, x)

                    base_frame = self._render_frame(x)
                # Side-view renders are read-only on gaussians — run outside
                # sim_lock so click events can acquire the lock sooner.
                left_frame = self._render_side_frame(x, side="left")
                right_frame = self._render_side_frame(x, side="right")

                self._update_controller_history()
                frame = self._draw_controller_markers(base_frame, draw_trail=True)

                do_auto_save = False
                with self.rec_lock:
                    if self.recording:
                        now = time.monotonic()
                        interval = 1.0 / self.rec_fps
                        if now - self._rec_last_time >= interval:
                            self._rec_last_time = now
                            if self.rec_no_trail:
                                rec_frame = self._draw_controller_markers(base_frame, draw_trail=False)
                            else:
                                rec_frame = frame
                            self.recorded_frames.append(rec_frame.copy())
                            self.recorded_particles.append(x.detach().cpu().numpy().copy())
                            packed_for_save = self._controllers_for_save()
                            self.recorded_controllers.append(packed_for_save)
                            if self.rec_max_frames > 0 and len(self.recorded_frames) >= self.rec_max_frames:
                                self.recording = False
                                do_auto_save = True

                frame = draw_status_overlay(
                    frame,
                    self.pressed_keys,
                    self.step_idx,
                    self.recording,
                    len(self.recorded_frames) / self.rec_fps if self.rec_fps > 0 else 0.0,
                    self._slot_summary_text(),
                    self.display_fps,
                )

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                for side_q, side_f, attr in [
                    (self.left_frame_queue, left_frame, "last_left_frame"),
                    (self.right_frame_queue, right_frame, "last_right_frame"),
                ]:
                    try:
                        side_q.put_nowait(side_f)
                    except queue.Full:
                        try:
                            side_q.get_nowait()
                            side_q.put_nowait(side_f)
                        except queue.Empty:
                            pass
                    setattr(self, attr, side_f)

                if do_auto_save:
                    self.stop_recording_and_save("", "")

                prev_x = x.clone()
                self.last_particles = x.clone()
                state["current_target"] = current_target
                state["prev_target"] = prev_target
                state["prev_x"] = prev_x
                self.step_idx += 1
                now = time.monotonic()
                if self._last_loop_end is not None:
                    dt = max(now - self._last_loop_end, 1e-6)
                    inst_fps = 1.0 / dt
                    if self.display_fps <= 0:
                        self.display_fps = inst_fps
                    else:
                        self.display_fps = 0.9 * self.display_fps + 0.1 * inst_fps
                self._last_loop_end = now
                time.sleep(0.001)
            except Exception as exc:
                print(f"[interactive_mpm] simulation loop error: {exc}")
                import traceback

                traceback.print_exc()
                break

    def get_frame(self):
        latest_frame = self.last_frame
        try:
            while True:
                frame = self.frame_queue.get_nowait()
                if isinstance(frame, np.ndarray):
                    latest_frame = frame
        except queue.Empty:
            pass

        if isinstance(latest_frame, np.ndarray):
            self.last_frame = latest_frame
            return latest_frame
        return self.last_frame

    def _drain_side_queue(self, q, last_attr):
        latest = getattr(self, last_attr)
        try:
            while True:
                f = q.get_nowait()
                if isinstance(f, np.ndarray):
                    latest = f
        except queue.Empty:
            pass
        if isinstance(latest, np.ndarray):
            setattr(self, last_attr, latest)
            return latest
        return getattr(self, last_attr)

    def get_left_frame(self):
        return self._drain_side_queue(self.left_frame_queue, "last_left_frame")

    def get_right_frame(self):
        return self._drain_side_queue(self.right_frame_queue, "last_right_frame")

    def _current_pivot(self):
        """Return the object centre in the original (rendering) frame, or None."""
        state = getattr(self, "sim_state", None)
        if not state:
            return None
        x = state.get("prev_x")
        if x is None:
            x = state.get("init_pos")
        if x is None:
            return None
        try:
            x_orig = self._to_original_frame(x)
            return x_orig.mean(dim=0).detach().cpu().numpy().astype(np.float64)
        except Exception:
            return None

    def set_novel_view_angle(self, side, angle_deg, axis="z"):
        """Update the camera angle/axis for a novel-view panel (left or right)."""
        angle_deg = float(angle_deg)
        axis = str(axis).lower()
        if axis not in ("x", "y", "z"):
            axis = "z"
        with self.sim_lock:
            device = str(self.trainer.device)
            pivot = self._current_pivot()
            new_w2c = orbit_w2c(self.w2c, angle_deg, axis=axis, pivot=pivot)
            new_view = build_gs_view(new_w2c, self.intrinsic, self.height, self.width, device=device)
            if side == "left":
                self.sim_state["left_view"] = new_view
                self.sim_state["left_w2c"] = new_w2c
            else:
                self.sim_state["right_view"] = new_view
                self.sim_state["right_w2c"] = new_w2c

    def get_status_html(self):
        """Return the current playground status as an HTML string for the side panel."""
        return build_status_text(
            self.pressed_keys,
            self.step_idx,
            self.recording,
            len(self.recorded_frames) / self.rec_fps if self.rec_fps > 0 else 0.0,
            self._slot_summary_text(),
            self.display_fps,
        )

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.setup_simulation()
            self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.sim_thread.start()
            return "Simulation started"
        return "Simulation already running"

    def stop_simulation(self):
        self.running = False
        self.display_fps = 0.0
        self._last_loop_end = None
        self.control_queue.put("stop")
        if self.sim_thread:
            self.sim_thread.join(timeout=2.0)
        return "Simulation stopped"

    def reset_simulation(self):
        if not self.running:
            return "Simulation not running"
        self.control_queue.put("reset")
        return "Reset requested"

    def send_key(self, key):
        if self.running and self.control_mode == "keyboard":
            self.control_queue.put(f"key:{key}")

    def release_key(self, key):
        if self.running and self.control_mode == "keyboard":
            self.control_queue.put(f"release:{key}")

    def start_recording(self):
        with self.rec_lock:
            self.recorded_frames.clear()
            self.recorded_particles.clear()
            self.recorded_controllers.clear()
            self._rec_last_time = 0.0
            self.recording = True
        if self.rec_max_frames > 0:
            max_sec = self.rec_max_frames / self.rec_fps
            return f"Recording... max {self.rec_max_frames} frames / {max_sec:.1f}s"
        return "Recording... no frame limit"

    def stop_recording_and_save(self, case_name: str, caption: str):
        with self.rec_lock:
            self.recording = False
            frames = list(self.recorded_frames)
            particles = list(self.recorded_particles)
            controllers = list(self.recorded_controllers)

        if len(frames) < 2:
            return "Need at least 2 frames."

        if not case_name.strip():
            case_name = f"rec_{self.record_count:04d}"
        self.record_count += 1

        case_dir = os.path.join(self.output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        video_path = os.path.join(case_dir, "video.mp4")
        writer = imageio.get_writer(
            video_path,
            fps=int(self.rec_fps),
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        ctrl_array = np.stack(controllers, axis=0).astype(np.float32)
        np.save(os.path.join(case_dir, "controller.npy"), ctrl_array)

        with open(os.path.join(case_dir, "calibrate.pkl"), "wb") as f:
            pickle.dump([np.linalg.inv(self.w2c)], f)
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump({"intrinsics": [self.intrinsic.tolist()], "WH": [self.width, self.height]}, f)

        status_msg = self._compute_and_save_particle_flow(
            particles,
            os.path.join(case_dir, "flow.npy"),
            self.intrinsic,
            self.w2c,
            self.width,
            self.height,
        )

        if not caption.strip():
            caption = f"A hand interacts with a deformable object ({case_name})."
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "metadata.jsonl"), "a") as f:
            f.write(json.dumps({"case": case_name, "caption": caption}) + "\n")

        return (
            f"Saved {len(frames)} frames -> {case_dir}\n"
            f"video.mp4 controller.npy flow.npy calibrate.pkl metadata.json\n"
            f"{status_msg}"
        )

    @staticmethod
    def _compute_and_save_particle_flow(particles, dst_path, intrinsic, w2c, width, height, splat_radius=5.0):
        t_steps = len(particles)
        if t_steps < 2:
            return "Need >= 2 frames for flow."

        r_mat = w2c[:3, :3].astype(np.float64)
        t_vec = w2c[:3, 3].astype(np.float64)
        k_mat = np.asarray(intrinsic, dtype=np.float64)
        sigma = splat_radius / 2.0
        r_int = int(np.ceil(splat_radius))

        flows = np.zeros((t_steps - 1, height, width, 2), dtype=np.float32)
        for i in range(t_steps - 1):
            pts0 = np.asarray(particles[i], dtype=np.float64)
            pts1 = np.asarray(particles[i + 1], dtype=np.float64)

            cam0 = (r_mat @ pts0.T).T + t_vec
            cam1 = (r_mat @ pts1.T).T + t_vec
            valid = (cam0[:, 2] > 0.01) & (cam1[:, 2] > 0.01)
            cam0 = cam0[valid]
            cam1 = cam1[valid]
            if cam0.shape[0] == 0:
                continue

            proj0 = (k_mat @ cam0.T).T
            proj1 = (k_mat @ cam1.T).T
            uv0 = (proj0[:, :2] / proj0[:, 2:3]).astype(np.float32)
            uv1 = (proj1[:, :2] / proj1[:, 2:3]).astype(np.float32)
            duv = uv1 - uv0
            inv_depth = (1.0 / cam0[:, 2]).astype(np.float32)

            weight_map = np.zeros((height, width), dtype=np.float32)
            flow_accum = np.zeros((height, width, 2), dtype=np.float32)
            for j in range(cam0.shape[0]):
                cx, cy = float(uv0[j, 0]), float(uv0[j, 1])
                u0 = max(int(cx - r_int), 0)
                u1 = min(int(cx + r_int) + 1, width)
                v0 = max(int(cy - r_int), 0)
                v1 = min(int(cy + r_int) + 1, height)
                if u0 >= u1 or v0 >= v1:
                    continue
                yy, xx = np.mgrid[v0:v1, u0:u1]
                dist_sq = (xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2
                weight = np.exp(-dist_sq / (2.0 * sigma * sigma)) * inv_depth[j]
                flow_accum[v0:v1, u0:u1, 0] += weight * duv[j, 0]
                flow_accum[v0:v1, u0:u1, 1] += weight * duv[j, 1]
                weight_map[v0:v1, u0:u1] += weight

            mask = weight_map > 1e-8
            flow_accum[mask, 0] /= weight_map[mask]
            flow_accum[mask, 1] /= weight_map[mask]
            flows[i] = flow_accum

        np.save(dst_path, flows.astype(np.float16))
        return f"Particle flow: {flows.shape} (float16)"


def create_gradio_interface(playground):
    with gr.Blocks(title="DeformMaster Interactive Playground (MPM)") as demo:
        gr.Markdown(
            """
            <style>
            #interactive_image, #interactive_image * {
                -webkit-user-drag: none;
                user-drag: none;
                -webkit-user-select: none;
                user-select: none;
                touch-action: none;
            }
            </style>
            """
        )
        gr.Markdown("# DeformMaster Interactive Playground")
        gr.Markdown("Interactively control deformable objects with physics-consistent dynamics and 4D appearance fidelity.")
        gr.Markdown(
            "<span style='color:red;font-weight:bold;'>Usage:</span> "
            "Press **Start Simulation** → In **Mouse mode**, click the image to select controller points → "
            "Switch to **Keyboard mode** and press **Start Simulation** again → "
            "Use W/A/S/D/Q/E (or I/J/K/L/U/O for the second controller) to manipulate the object. "
            "Press **Reset** to re-pick controllers."
        )

        # --- Main row: NV1 side | Main View | NV2 side ---
        with gr.Row():
            with gr.Column(scale=2, min_width=180):
                left_image = gr.Image(label="Novel View 1", type="numpy", interactive=False)
                with gr.Group():
                    nv1_angle = gr.Slider(0, 120, value=30, step=1, label="Novel View 1 Angle (deg)")
                    nv1_axis = gr.Radio(
                        choices=["x", "y", "z"], value="z",
                        label="NV1 Axis", interactive=True,
                    )
                with gr.Group():
                    control_mode_selector = gr.Radio(
                        choices=["mouse", "keyboard"],
                        value=playground.control_mode,
                        label="Control Mode",
                        interactive=True,
                        elem_id="control_mode_selector",
                    )
                    slot_choices = [str(i + 1) for i in range(playground.n_ctrl_parts)]
                    slot_selector = gr.Radio(
                        choices=slot_choices,
                        value=slot_choices[0],
                        label="Mouse Pick Controller",
                        interactive=True,
                        elem_id="slot_selector",
                    )
                    inv_ctrl_checkbox = gr.Checkbox(
                        value=playground.inv_ctrl,
                        label="Invert Control Direction",
                        interactive=True,
                    )
            with gr.Column(scale=4):
                image_output = gr.Image(label="Interactive Playground", type="numpy", elem_id="interactive_image")
                with gr.Row():
                    start_btn = gr.Button("Start Simulation", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
                    stop_btn = gr.Button("Stop Simulation", variant="stop")
                status_text = gr.Textbox(label="Status", value="Not started", interactive=False)
            with gr.Column(scale=2, min_width=180):
                right_image = gr.Image(label="Novel View 2", type="numpy", interactive=False)
                with gr.Group():
                    nv2_angle = gr.Slider(0, 120, value=30, step=1, label="Novel View 2 Angle (deg)")
                    nv2_axis = gr.Radio(
                        choices=["x", "y", "z"], value="z",
                        label="NV2 Axis", interactive=True,
                    )
                with gr.Group():
                    live_status_html = gr.HTML(
                        value=build_status_text(set(), 0, False, 0.0, "-", 0.0),
                        label="Live Status",
                    )

        # --- Bottom section: Material Controls | Control Panels ---
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Material Controls")
                with gr.Row():
                    e_scale_slider = gr.Slider(0.2, 5.0, value=1.0, step=0.05, label="Young's Modulus Scale")
                    nu_delta_slider = gr.Slider(-0.20, 0.20, value=0.0, step=0.01, label="Poisson Ratio Delta")
                with gr.Row():
                    ctrl_stiff_slider = gr.Slider(0.1, 10.0, value=10.0, step=0.05, label="Controller Stiffness Scale")
                    ctrl_damp_slider = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Controller Damping Scale")
                with gr.Row():
                    _cfg_steps_default = int(playground.trainer.cfg.mpm.steps_per_frame)
                    _cfg_dt = float(playground.trainer.cfg.mpm.dt)
                    substep_slider = gr.Slider(
                        4, max(60, _cfg_steps_default), value=_cfg_steps_default,
                        step=1,
                        label=f"Sim substeps / frame (default {_cfg_steps_default}; lower = faster wall-clock, slower physics)",
                    )
                    substep_readout = gr.Markdown(
                        f"Physical Δt / frame = "
                        f"`{_cfg_dt * _cfg_steps_default * 1000:.2f} ms`"
                    )
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("### Control Panel 1")
                with gr.Row():
                    btn_q = gr.Button("Q", min_width=40)
                    btn_w = gr.Button("W", min_width=40)
                    btn_e = gr.Button("E", min_width=40)
                with gr.Row():
                    btn_a = gr.Button("A", min_width=40)
                    btn_s = gr.Button("S", min_width=40)
                    btn_d = gr.Button("D", min_width=40)
            if playground.n_ctrl_parts > 1:
                with gr.Column(scale=1, min_width=140):
                    gr.Markdown("### Control Panel 2")
                    with gr.Row():
                        btn_u = gr.Button("U", min_width=40)
                        btn_i = gr.Button("I", min_width=40)
                        btn_o = gr.Button("O", min_width=40)
                    with gr.Row():
                        btn_j = gr.Button("J", min_width=40)
                        btn_k = gr.Button("K", min_width=40)
                        btn_l = gr.Button("L", min_width=40)

        def start_sim():
            result = playground.start_simulation()
            return result, result

        def stop_sim():
            result = playground.stop_simulation()
            return "Stopped", result

        def _placeholder():
            ph = np.zeros((playground.height, playground.width, 3), dtype=np.uint8)
            ph[:, :] = [30, 30, 30]
            return ph

        def update_frame():
            frame = playground.get_frame()
            if frame is not None:
                return frame
            if not playground.running:
                ph = _placeholder()
                cv2.putText(
                    ph,
                    "Click 'Start Simulation' to begin",
                    (playground.width // 6, playground.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                return ph
            return None

        def update_left_frame():
            f = playground.get_left_frame()
            return f if f is not None else _placeholder()

        def update_right_frame():
            f = playground.get_right_frame()
            return f if f is not None else _placeholder()

        def reset_sim():
            return playground.reset_simulation()

        def on_control_mode_change(mode):
            return playground.set_control_mode(mode)

        def on_material_change(e_scale, nu_delta, ctrl_stiffness_scale, ctrl_damping_scale):
            return playground.set_material_controls(
                e_scale,
                nu_delta,
                1.0,
                ctrl_stiffness_scale,
                ctrl_damping_scale,
            )

        def on_image_select(slot_value, evt: gr.SelectData):
            if evt is None or evt.index is None:
                return "No click received."
            x_pix, y_pix = evt.index
            return playground.select_controller_from_click(x_pix, y_pix, int(slot_value) - 1)

        def make_key_handler(key, duration=0.1):
            def press():
                playground.send_key(key)

                def auto_release():
                    time.sleep(duration)
                    playground.release_key(key)

                threading.Thread(target=auto_release, daemon=True).start()
                return f"Key {key.upper()} pressed"

            return press

        start_btn.click(start_sim, outputs=[status_text, status_text])
        reset_btn.click(reset_sim, outputs=[status_text])
        stop_btn.click(stop_sim, outputs=[status_text, status_text])
        control_mode_selector.change(on_control_mode_change, inputs=[control_mode_selector], outputs=[status_text])
        inv_ctrl_checkbox.change(lambda v: playground.set_inv_ctrl(v), inputs=[inv_ctrl_checkbox], outputs=[status_text])
        material_inputs = [
            e_scale_slider,
            nu_delta_slider,
            ctrl_stiff_slider,
            ctrl_damp_slider,
        ]
        e_scale_slider.release(on_material_change, inputs=material_inputs, outputs=[status_text])
        nu_delta_slider.release(on_material_change, inputs=material_inputs, outputs=[status_text])
        ctrl_stiff_slider.release(on_material_change, inputs=material_inputs, outputs=[status_text])
        ctrl_damp_slider.release(on_material_change, inputs=material_inputs, outputs=[status_text])

        def on_substep_change(n):
            n_int = max(1, int(n))
            playground.steps_per_frame_override = n_int
            return f"Physical Δt / frame = `{_cfg_dt * n_int * 1000:.2f} ms`"
        substep_slider.release(on_substep_change,
                               inputs=[substep_slider],
                               outputs=[substep_readout])

        def on_nv1_change(angle, axis):
            playground.set_novel_view_angle("left", float(angle), axis)

        def on_nv2_change(angle, axis):
            playground.set_novel_view_angle("right", -float(angle), axis)

        nv1_angle.release(on_nv1_change, inputs=[nv1_angle, nv1_axis])
        nv1_axis.change(on_nv1_change, inputs=[nv1_angle, nv1_axis])
        nv2_angle.release(on_nv2_change, inputs=[nv2_angle, nv2_axis])
        nv2_axis.change(on_nv2_change, inputs=[nv2_angle, nv2_axis])

        btn_w.click(make_key_handler("w"), outputs=[status_text])
        btn_s.click(make_key_handler("s"), outputs=[status_text])
        btn_a.click(make_key_handler("a"), outputs=[status_text])
        btn_d.click(make_key_handler("d"), outputs=[status_text])
        btn_q.click(make_key_handler("q"), outputs=[status_text])
        btn_e.click(make_key_handler("e"), outputs=[status_text])

        if playground.n_ctrl_parts > 1:
            btn_i.click(make_key_handler("i"), outputs=[status_text])
            btn_k.click(make_key_handler("k"), outputs=[status_text])
            btn_j.click(make_key_handler("j"), outputs=[status_text])
            btn_l.click(make_key_handler("l"), outputs=[status_text])
            btn_u.click(make_key_handler("u"), outputs=[status_text])
            btn_o.click(make_key_handler("o"), outputs=[status_text])

        gr.Markdown("---")
        gr.Markdown("### Data Collection")
        with gr.Row():
            case_name_input = gr.Textbox(label="Case name (auto-generated if empty)")
            caption_input = gr.Textbox(label="Caption / text description")
        with gr.Row():
            rec_start_btn = gr.Button("Start Recording", variant="primary")
            rec_stop_btn = gr.Button("Stop & Save", variant="stop")
            rec_frames_text = gr.Textbox(label="Recorded frames", value="0", interactive=False)
        rec_result_text = gr.Textbox(label="Save result", interactive=False, lines=3)

        def on_rec_start():
            msg = playground.start_recording()
            return msg, "0"

        def on_rec_stop(case_name, caption):
            return playground.stop_recording_and_save(case_name, caption)

        def update_rec_frames():
            n = len(playground.recorded_frames)
            secs = n / playground.rec_fps if playground.rec_fps > 0 else 0
            if playground.recording:
                return f"{n} frames ({secs:.1f}s)"
            return f"{n} frames ({secs:.1f}s) - stopped"

        rec_start_btn.click(on_rec_start, outputs=[rec_result_text, rec_frames_text])
        rec_stop_btn.click(on_rec_stop, inputs=[case_name_input, caption_input], outputs=[rec_result_text])

        keyboard_js = """
        () => {
            const GAME_KEYS = new Set(['w','a','s','d','q','e','i','j','k','l','u','o']);
            const held = new Set();

            const getControlMode = () => {
                const root = document.getElementById('control_mode_selector');
                if (!root) return 'mouse';
                const checked = root.querySelector('input[type="radio"]:checked');
                return checked ? checked.value : 'mouse';
            };

            const attachImageGuards = () => {
                const root = document.getElementById('interactive_image');
                if (!root || root.dataset.dragBound === '1') return;
                root.dataset.dragBound = '1';

                const killNativeDrag = (el) => {
                    if (!el) return;
                    el.setAttribute('draggable', 'false');
                    el.draggable = false;
                    el.ondragstart = () => false;
                    el.addEventListener('dragstart', (evt) => {
                        evt.preventDefault();
                        evt.stopPropagation();
                        return false;
                    });
                };

                killNativeDrag(root);
                root.querySelectorAll('*').forEach(killNativeDrag);
            };

            document.addEventListener('keydown', (e) => {
                if (getControlMode() !== 'keyboard') return;
                const tag = e.target.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA') return;
                const k = e.key.toLowerCase();
                if (k === 'r') { e.preventDefault(); fetch('/api/reset', {method:'POST'}); return; }
                if (e.key === ' ') { e.preventDefault(); fetch('/api/rec_toggle', {method:'POST'}); return; }
                if (!GAME_KEYS.has(k)) return;
                e.preventDefault();
                if (held.has(k)) return;
                held.add(k);
                fetch('/api/key/down/' + k, {method: 'POST'});
            });
            document.addEventListener('keyup', (e) => {
                if (getControlMode() !== 'keyboard') return;
                const k = e.key.toLowerCase();
                if (!GAME_KEYS.has(k)) return;
                e.preventDefault();
                held.delete(k);
                fetch('/api/key/up/' + k, {method: 'POST'});
            });

            document.addEventListener('dragstart', (e) => {
                const root = document.getElementById('interactive_image');
                if (root && root.contains(e.target)) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }
            }, true);

            setTimeout(attachImageGuards, 500);
            setTimeout(attachImageGuards, 1500);
            setTimeout(attachImageGuards, 3000);
        }
        """

        demo.load(fn=update_frame, inputs=None, outputs=image_output, js=keyboard_js)
        # Click events get their own concurrency lane so they don't queue behind
        # the 30fps timer ticks.
        image_output.select(
            on_image_select,
            inputs=[slot_selector],
            outputs=[status_text],
            concurrency_id="click",
            concurrency_limit=4,
        )

        timer_comp = gr.Timer(value=0.033, active=True)

        def update_frame_and_counter():
            return (
                update_frame(),
                update_left_frame(),
                update_right_frame(),
                update_rec_frames(),
                playground.get_status_html(),
            )

        # Timer ticks also get their own lane (separate from click lane).
        timer_comp.tick(
            fn=update_frame_and_counter,
            inputs=None,
            outputs=[image_output, left_image, right_image, rec_frames_text, live_status_html],
            concurrency_id="ui_tick",
            concurrency_limit=4,
        )

    return demo


CATEGORY_DIRS = (
    "softbody_warp",
    "cloth_warp",
    "rope_warp",
    "package_warp",
    "softbody_warp_finetune",
    "cloth_warp_finetune",
    "rope_warp_finetune",
    "package_warp_finetune",
)


def resolve_from_output(output, case_name):
    """Auto-resolve config.yaml and best_checkpoint.pt from output directory."""
    candidates = [
        os.path.join(output, case_name),
        output,
    ]
    for cat in CATEGORY_DIRS:
        candidates.append(os.path.join(output, cat, case_name))

    for scene_dir in candidates:
        config_path = os.path.join(scene_dir, "config.yaml")
        ckpt_path = os.path.join(scene_dir, "best_checkpoint.pt")
        if os.path.exists(config_path) and os.path.exists(ckpt_path):
            return config_path, ckpt_path

    raise FileNotFoundError(
        f"Cannot find config.yaml + best_checkpoint.pt for '{case_name}' under '{output}'. "
        f"Searched direct case dir, direct output dir, and categories: {', '.join(CATEGORY_DIRS)}"
    )


def pick_listen_port(host, preferred, span=50):
    bind_host = "" if host in ("0.0.0.0", "::") else host
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in [{preferred}, {preferred + span})")


def main():
    parser = ArgumentParser()
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="Training output dir (e.g. output_4). Auto-resolves config and checkpoint.")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default=None,
                        help="Explicit path to the main-view background image. "
                             "If unset, falls back to ./data/bg_mono.jpg when --bg_mono "
                             "is passed, otherwise ./data/bg.png.")
    parser.add_argument("--bg_mono", action="store_true",
                        help="Use the mono-capture background (./data/bg_mono.jpg) "
                             "instead of the default multi-cam background (./data/bg.png).")
    parser.add_argument("--bg_blank", action="store_true",
                        help="Use a solid blank (white) background instead of any image. "
                             "Overrides --bg_img_path / --bg_mono.")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument("--inv_ctrl", action="store_true", help="Enable inverted control directions (default: off)")
    parser.add_argument("--output_dir", type=str, default="playground_recording/physics_flow")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle_iters", type=int, default=220)
    parser.add_argument("--control_mode", type=str, default="mouse", choices=["mouse", "keyboard"])
    parser.add_argument("--rec_max_frames", type=int, default=0, help="0 means unlimited recording length")
    parser.add_argument(
        "--rec_no_trail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide controller history trajectory (trail) in the saved recording video. Use --no-rec_no_trail to keep the trail. The live display and current-position marker are unaffected.",
    )
    args = parser.parse_args()
    if args.case_name == "my_mono_cloth" and args.bg_img_path is None and not args.bg_blank:
        args.bg_mono = True

    set_all_seeds(args.seed)

    resolved_config_path, ckpt_path = resolve_from_output(args.output, args.case_name)
    print(f"Config: {resolved_config_path}")
    print(f"Checkpoint: {ckpt_path}")
    cfg = OmegaConf.load(resolved_config_path)
    cfg.mpm.device = "cuda"

    with open(os.path.join(args.base_path, args.case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    with open(os.path.join(args.base_path, args.case_name, "metadata.json"), "r") as f:
        meta = json.load(f)

    width, height = meta["WH"]
    intrinsic = np.asarray(meta["intrinsics"][0], dtype=np.float32)
    w2c = np.asarray(w2cs[0], dtype=np.float32)

    if args.bg_blank:
        print("[playground] Main-view background: blank (white)")
        overlay = np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        if args.bg_img_path is not None:
            bg_path = args.bg_img_path
        elif args.bg_mono:
            bg_path = "./data/bg_mono.jpg"
        else:
            bg_path = "./data/bg.png"
        overlay = cv2.imread(bg_path)
        if overlay is None:
            raise FileNotFoundError(
                f"Background image not found: {bg_path} "
                f"(try --bg_img_path <path>, --bg_mono, or --bg_blank)"
            )
        print(f"[playground] Main-view background: {bg_path}")
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
    overlay = torch.tensor(overlay, dtype=torch.float32, device="cuda")

    gs_path = os.path.join(
        args.gaussian_path,
        args.case_name,
        EXP_NAME,
        "point_cloud",
        "iteration_10000",
        "point_cloud.ply",
    )
    if not os.path.exists(gs_path):
        raise FileNotFoundError(f"Gaussian model not found: {gs_path}")

    trainer = PlaygroundEngine(cfg, args.case_name)
    trainer.load_from_checkpoint(ckpt_path)

    playground = GradioInteractiveMPMPlayground(
        trainer=trainer,
        gs_path=gs_path,
        intrinsic=intrinsic,
        w2c=w2c,
        width=width,
        height=height,
        overlay_image=overlay,
        n_ctrl_parts=args.n_ctrl_parts,
        inv_ctrl=args.inv_ctrl,
        output_dir=args.output_dir,
        settle_iters=args.settle_iters,
        control_mode=args.control_mode,
        rec_max_frames=args.rec_max_frames,
        rec_no_trail=args.rec_no_trail,
    )

    demo = create_gradio_interface(playground)

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    fast_app = FastAPI()

    @fast_app.post("/api/key/down/{key}")
    async def key_down(key: str):
        if key in VALID_KEYS and playground.control_mode == "keyboard":
            playground.send_key(key)
        return JSONResponse({"ok": True})

    @fast_app.post("/api/key/up/{key}")
    async def key_up(key: str):
        if key in VALID_KEYS and playground.control_mode == "keyboard":
            playground.release_key(key)
        return JSONResponse({"ok": True})

    @fast_app.post("/api/reset")
    async def reset():
        playground.reset_simulation()
        return JSONResponse({"ok": True})

    @fast_app.post("/api/rec_toggle")
    async def rec_toggle():
        if playground.recording:
            result = playground.stop_recording_and_save("", "")
            return JSONResponse({"recording": False, "result": result})
        playground.start_recording()
        return JSONResponse({"recording": True})

    # Allow many concurrent events so timer ticks don't starve mouse clicks
    demo.queue(default_concurrency_limit=10)
    app = gr.mount_gradio_app(fast_app, demo, path="/")
    port = pick_listen_port(args.server_name, args.server_port)
    if port != args.server_port:
        print(f"[interactive_mpm] Port {args.server_port} is busy, switched to {port}")
    print(f"[interactive_mpm] Open http://127.0.0.1:{port}")
    uvicorn.run(app, host=args.server_name, port=port)


if __name__ == "__main__":
    main()
