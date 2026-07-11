"""Runtime backend for the interactive DeformMaster playground.

This module intentionally contains only the state loading and rollout setup
needed by ``interactive_playground_online.py``. Training loops, losses,
parameter updates, schedulers, and checkpoint writers are not part of the
public playground backend.
"""

import os
import pickle
import warnings

warnings.filterwarnings(
    "ignore",
    message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    module="warp",
)

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, sample_farthest_points

ti.init(arch=ti.cpu, log_level=ti.WARN)

from ..data.dataset_mpm import DeformMasterDataset
from ..model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator
from ..model.residual_pgnd import ResidualPGND
from ..utils.mpm_utils import youngs_poisson_to_lame


class PlaygroundEngine:
    """Checkpoint-loaded runtime state used by the online playground."""

    def __init__(self, cfg, scene_id):
        self.cfg = cfg
        self.device = torch.device(cfg.mpm.device)
        self.scene_id = scene_id
        self.best_loss = float("inf")
        self.loss_history = []
        self.resume_checkpoint = None

        self.dataset = DeformMasterDataset(cfg, case_name=scene_id)
        if len(self.dataset) == 0:
            raise ValueError(f"Scene {scene_id} not found!")
        self.data = self.dataset[0]

        self._apply_controller_mask()
        self._init_parameters()
        self._init_simulator()
        self._init_residual()
        self._apply_mono_alignment()
        self._init_scene_state()

    def _apply_controller_mask(self):
        data_root = getattr(self.cfg.data, "root", "./data/different_types")
        mask_path = os.path.join(data_root, self.scene_id, "controller_mask.npy")
        cfg_keep_n = getattr(self.cfg.mpm, "controller_keep_n", None)
        mask_disabled = cfg_keep_n is not None and cfg_keep_n <= 0

        cp = self.data["controller_points"]
        self._raw_ctrl_frame0 = cp[0].clone() if isinstance(cp, torch.Tensor) else cp[0].copy()
        self._controller_mask_used = None

        if mask_disabled:
            print(f"[INFO] Controller mask disabled (controller_keep_n={cfg_keep_n}) for {self.scene_id}")
        elif os.path.exists(mask_path):
            ctrl_mask = np.load(mask_path).astype(bool)
            self._controller_mask_used = ctrl_mask
            if isinstance(cp, torch.Tensor):
                mask_t = torch.from_numpy(ctrl_mask).to(cp.device)
                self.data["controller_points"] = cp[:, mask_t]
            else:
                self.data["controller_points"] = cp[:, ctrl_mask]
            print(f"[INFO] Applied controller mask for {self.scene_id}: {int(ctrl_mask.sum())}/{len(ctrl_mask)}")

    def _init_parameters(self):
        self.n_patches = getattr(self.cfg.mpm, "n_patches", 64)
        self.active_experts = getattr(self.cfg.mpm, "active_experts", ["nh", "co", "st", "fi"])
        num_experts = len(self.active_experts)

        config_weights = getattr(self.cfg.mpm, "init_weights", None)
        if config_weights and len(config_weights) == num_experts:
            w_tensor = torch.tensor(config_weights, device=self.device)
            w_tensor = w_tensor / w_tensor.sum()
            init_log_weights = torch.log(w_tensor + 1e-6)
            self.log_weights = nn.Parameter(init_log_weights.unsqueeze(0).repeat(self.n_patches, 1))
            print(f"Initialized expert weights from config: {config_weights}")
        else:
            init_weights = torch.ones(self.n_patches, num_experts, device=self.device) / num_experts
            self.log_weights = nn.Parameter(torch.log(init_weights))
            print("Initialized expert weights uniformly.")

        init_val = getattr(self.cfg.mpm, "init_raw_params", 0.0)

        def get_init(name, default):
            return getattr(self.cfg.mpm, f"init_raw_{name}", default)

        self.raw_E = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init("E", init_val))
        self.raw_nu = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init("nu", init_val))
        self.raw_fiber_k = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init("fiber_k", init_val))
        self.raw_yield = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init("yield", init_val))
        self.raw_viscosity = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init("viscosity", init_val))
        self.raw_fiber_dir = nn.Parameter(torch.randn(self.n_patches, 3, device=self.device) * 0.1)
        self.raw_ctrl_stiffness = nn.Parameter(
            torch.tensor([get_init("ctrl_stiffness", 0.0)], device=self.device)
        )
        self.raw_ctrl_damping = nn.Parameter(torch.tensor([get_init("ctrl_damping", 0.0)], device=self.device))

    def _init_simulator(self):
        self.use_warp = getattr(self.cfg.mpm, "use_warp", True)
        if not self.use_warp:
            raise ValueError("PyTorch MPM backend has been removed; set mpm.use_warp: true.")
        print("[INFO] Using NVIDIA Warp backend for MPM simulation.")
        self.simulator = WarpMPMSimulator(self.cfg.mpm).to(self.device)
        self.simulator.debug_mode = True

    def _init_residual(self):
        residual_cfg = getattr(self.cfg, "residual", None)
        residual_enabled = residual_cfg is not None and getattr(residual_cfg, "enabled", True)
        if residual_enabled:
            self.residual_cfg = residual_cfg
            arch = getattr(residual_cfg, "arch", "pgnd")
            if arch != "pgnd":
                raise ValueError(
                    f"Unsupported residual arch {arch!r} in the release version; only 'pgnd' is included."
                )
            self.residual_net = ResidualPGND(residual_cfg).to(self.device)
            print(f"[INFO] Residual model initialized: arch={arch}")
        else:
            self.residual_cfg = None
            self.residual_net = None
            reason = "disabled via enabled=false" if residual_cfg is not None else "no residual config"
            print(f"[INFO] ResidualPGND disabled ({reason}).")

    def _apply_mono_alignment(self):
        if not getattr(self.cfg.data, "mono_align", False):
            self._mono_align_R = None
            return

        init_raw = self.data["init_pos"].to(self.device)
        centered = init_raw - init_raw.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if normal[1].item() < 0:
            normal = -normal
        target = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=normal.dtype)
        v = torch.linalg.cross(normal, target)
        s = torch.linalg.norm(v)
        c = torch.dot(normal, target)
        if s.item() < 1e-6:
            rotation = (
                torch.eye(3, device=self.device, dtype=normal.dtype)
                if c > 0
                else torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=normal.dtype))
            )
        else:
            vx = torch.tensor(
                [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
                device=self.device,
                dtype=normal.dtype,
            )
            rotation = torch.eye(3, device=self.device, dtype=normal.dtype) + vx + vx @ vx * (
                (1.0 - c) / (s * s)
            )

        self._mono_align_R = rotation
        for key in ["init_pos", "gt_surface_tracks", "controller_points"]:
            if key in self.data:
                self.data[key] = self.data[key].to(self.device) @ rotation.T

        init_pts = self.data["init_pos"]
        ip_min = init_pts.min(dim=0)[0]
        ip_max = init_pts.max(dim=0)[0]
        print(f"[mono_align] plane normal={normal.tolist()}, rotation det={torch.linalg.det(rotation).item():.4f}")
        print(
            f"[mono_align] init_pos bbox after rotate: "
            f"X[{ip_min[0]:.3f},{ip_max[0]:.3f}] "
            f"Y[{ip_min[1]:.3f},{ip_max[1]:.3f}] "
            f"Z[{ip_min[2]:.3f},{ip_max[2]:.3f}], "
            f"max_span={(ip_max - ip_min).max().item():.3f}"
        )

    def _init_scene_state(self):
        obj_pts = self.data["init_pos"].to(self.device)
        gt_pts = self.data["gt_surface_tracks"].to(self.device).view(-1, 3)
        ctrl_pts = self.data["controller_points"].to(self.device).view(-1, 3)

        gt_pts = gt_pts[torch.linalg.norm(gt_pts, dim=-1) > 1e-5]
        ctrl_pts = ctrl_pts[torch.linalg.norm(ctrl_pts, dim=-1) > 1e-5]

        all_pts = torch.cat([obj_pts, gt_pts, ctrl_pts], dim=0)
        p_min = all_pts.min(dim=0)[0]
        p_max = all_pts.max(dim=0)[0]
        floor_margin = getattr(self.cfg.mpm, "floor_margin", 0.05)
        init_min_z = obj_pts[:, 2].min()
        self.auto_offset = torch.zeros(3, device=self.device)
        self.auto_offset[0] = -(p_min[0] + p_max[0]) / 2.0
        self.auto_offset[1] = -(p_min[1] + p_max[1]) / 2.0
        self.auto_offset[2] = floor_margin - init_min_z

        self.simulator.base_offset = self.auto_offset
        self.simulator._apply_boundary()
        self.simulator.reset(obj_pts, controller_pos=self.data["controller_points"][0].to(self.device))

        self.controller_points = self.data["controller_points"].to(self.device) + self.auto_offset
        window = getattr(self.cfg.mpm, "controller_smooth_window", 5)
        if window > 1:
            smoothed = self.controller_points.clone()
            t_ctrl = self.controller_points.shape[0]
            for t in range(t_ctrl):
                t_start = max(0, t - window // 2)
                t_end = min(t_ctrl, t + window // 2 + 1)
                smoothed[t] = self.controller_points[t_start:t_end].mean(dim=0)
            self.controller_points = smoothed
            print(f"[INFO] Controller trajectory smoothed (window={window})")

        xyz_static = self.data["gaussians"][:, :3].unsqueeze(0).to(self.device)
        self.patch_centers, _ = sample_farthest_points(xyz_static, K=self.n_patches)
        self._refresh_patch_assignment()

        if "fi" in self.active_experts:
            self._init_fiber_dir_from_geometry((self.data["init_pos"].to(self.device) + self.auto_offset))

    def _refresh_patch_assignment(self):
        init_pos_centered = (self.data["init_pos"].to(self.device) + self.auto_offset).unsqueeze(0)
        dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
        dist = torch.clamp(dist, min=1e-6)
        inv_dist = 1.0 / dist
        norm = torch.sum(inv_dist, dim=2, keepdim=True)
        self.interp_weights = (inv_dist / norm).unsqueeze(-1)

    def _init_fiber_dir_from_geometry(self, particle_pos):
        pca_k = min(16, particle_pos.shape[0])
        _, nbr_idx, _ = knn_points(self.patch_centers, particle_pos.unsqueeze(0), K=pca_k)
        nbr_pos = particle_pos[nbr_idx.squeeze(0)]
        centered = nbr_pos - nbr_pos.mean(dim=1, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        tangent = vh[:, 0, :]
        with torch.no_grad():
            self.raw_fiber_dir.copy_(tangent)
        print(f"[INFO] Fiber directions initialized from local PCA (k={pca_k})")

    def get_current_phys_props(self):
        weights = torch.softmax(self.log_weights, dim=-1)
        bounds = self.cfg.mpm.get("material_bounds", {})
        friction = torch.tensor(self.cfg.mpm.boundary.friction, device=self.device)

        e_min, e_max = bounds.get("E", [1e3, 1e6])
        nu_min, nu_max = bounds.get("nu", [0.0, 0.45])
        fk_min, fk_max = bounds.get("fiber_k", [5e3, 5e5])
        ys_min, ys_max = bounds.get("yield_stress", [1e2, 1e5])
        visc_min, visc_max = bounds.get("plastic_viscosity", [0.1, 10.0])

        def map_log_sigmoid(raw_param, min_val, max_val):
            log_min = np.log(max(min_val, 1e-6))
            log_max = np.log(max_val)
            log_val = log_min + torch.sigmoid(raw_param) * (log_max - log_min)
            return torch.exp(log_val)

        p_e = map_log_sigmoid(self.raw_E, e_min, e_max)
        p_nu = nu_min + torch.sigmoid(self.raw_nu) * (nu_max - nu_min)
        p_mu, p_lam = youngs_poisson_to_lame(p_e, p_nu)
        p_fiber_k = map_log_sigmoid(self.raw_fiber_k, fk_min, fk_max)
        p_yield = map_log_sigmoid(self.raw_yield, ys_min, ys_max)
        p_visc = map_log_sigmoid(self.raw_viscosity, visc_min, visc_max)

        cs_min, cs_max = bounds.get("controller_stiffness", [1e3, 1e6])
        cd_min, cd_max = bounds.get("controller_damping", [1e2, 5e4])
        p_ctrl_stiffness = map_log_sigmoid(self.raw_ctrl_stiffness, cs_min, cs_max)
        p_ctrl_damping = map_log_sigmoid(self.raw_ctrl_damping, cd_min, cd_max)

        return (
            weights,
            p_mu.squeeze(),
            p_lam.squeeze(),
            p_fiber_k.squeeze(),
            self.raw_fiber_dir,
            friction,
            p_yield.squeeze(),
            p_e.squeeze(),
            p_nu.squeeze(),
            p_visc.squeeze(),
            p_ctrl_stiffness.squeeze(),
            p_ctrl_damping.squeeze(),
        )

    def load_from_checkpoint(self, resume_path):
        print(f"[RESUME] Loading state from {resume_path}...")
        if resume_path.endswith(".pt"):
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            state = checkpoint["model_state_dict"]
            with torch.no_grad():
                self.log_weights.copy_(state["log_weights"])
                self.raw_E.copy_(state["raw_E"])
                self.raw_nu.copy_(state["raw_nu"])
                self.raw_fiber_k.copy_(state["raw_fiber_k"])
                self.raw_fiber_dir.copy_(state["raw_fiber_dir"])
                self.raw_yield.copy_(state["raw_yield"])
                self.raw_viscosity.copy_(state["raw_viscosity"])
                if "raw_ctrl_stiffness" in state:
                    self.raw_ctrl_stiffness.copy_(state["raw_ctrl_stiffness"])
                if "raw_ctrl_damping" in state:
                    self.raw_ctrl_damping.copy_(state["raw_ctrl_damping"])

                if "best_loss" in checkpoint:
                    self.best_loss = checkpoint["best_loss"]
                    print(f"[RESUME] Best loss restored: {self.best_loss:.6f}")
                if "loss_history" in checkpoint:
                    self.loss_history = checkpoint["loss_history"]
                    print(f"[RESUME] Loss history restored ({len(self.loss_history)} iters).")

                if "auto_offset" in state:
                    self.auto_offset = state["auto_offset"].to(self.device)
                    self.simulator.base_offset = self.auto_offset
                    self.simulator._apply_boundary()
                    print("[RESUME] Auto-centering offset restored.")

                if "patch_centers" in state:
                    self.patch_centers = state["patch_centers"].to(self.device)
                    self._refresh_patch_assignment()
                    print("[RESUME] Persistent patch assignment restored.")

                if "residual_net" in state and self.residual_net is not None:
                    try:
                        self.residual_net.load_state_dict(state["residual_net"])
                        print("[RESUME] ResidualPGND weights loaded.")
                    except Exception as exc:
                        print(f"[RESUME] Warning: Failed to load ResidualPGND weights: {exc}")
            self.resume_checkpoint = checkpoint
            return

        with open(resume_path, "rb") as f:
            checkpoint = pickle.load(f)

        with torch.no_grad():
            if "raw_E" in checkpoint:
                self.raw_E.copy_(torch.from_numpy(checkpoint["raw_E"]).to(self.device))
            if "raw_nu" in checkpoint:
                self.raw_nu.copy_(torch.from_numpy(checkpoint["raw_nu"]).to(self.device))
            if "raw_fiber_k" in checkpoint:
                self.raw_fiber_k.copy_(torch.from_numpy(checkpoint["raw_fiber_k"]).to(self.device))
            if "raw_fiber_dir" in checkpoint:
                self.raw_fiber_dir.copy_(torch.from_numpy(checkpoint["raw_fiber_dir"]).to(self.device))
            if "raw_yield" in checkpoint:
                self.raw_yield.copy_(torch.from_numpy(checkpoint["raw_yield"]).to(self.device))
            if "raw_viscosity" in checkpoint:
                self.raw_viscosity.copy_(torch.from_numpy(checkpoint["raw_viscosity"]).to(self.device))
            if "log_weights" in checkpoint:
                self.log_weights.copy_(torch.from_numpy(checkpoint["log_weights"]).to(self.device))
        self.resume_checkpoint = None
