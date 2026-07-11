import torch
import torch.nn as nn
import warp as wp
import numpy as np
from .mpm_solver_warp import MPM_Simulator_WARP
from .warp_utils import (
    torch2warp_float, torch2warp_vec3, torch2warp_mat33,
    add_vec3_to_vec3,
    accumulate_pd_forces_kernel, apply_dv_with_clamping_kernel, set_vec3_to_zero
)

class WarpMPMSimulator(nn.Module):
    """
    NVIDIA Warp-based MPM Simulator for DeformMaster.
    Provides a similar interface to MPMSimulator but uses Warp backend.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.warp_device = "cuda:0" # Warp usually defaults to cuda:0
        
        # Initialize Warp
        wp.init()
        wp.config.verify_cuda = True
        wp.config.quiet = True # [FIX] Suppress Warp JIT loading messages to keep terminal clean
        
        self.n_particles = int(cfg.n_particles)
        self.grid_res = int(cfg.grid_res)
        self.dt = float(cfg.dt)
        
        # Internal Warp state
        self.solver = MPM_Simulator_WARP(self.n_particles, n_grid=self.grid_res, device=self.warp_device)
        self.tape = None
        
        # Temp buffers for kernel outputs
        self.particle_dv_wp = None # Allocated in reset
        
        # Shared shift for grid alignment
        self.shift = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        
        # Controller state
        self.controller_indices_wp = None
        self.controller_mask_wp = None
        self.initial_offsets_wp = None
        self.count_per_particle_wp = None
        self.num_connections = 0

    def reset(self, particles, volume=None, controller_pos=None):
        """
        Reset simulation state with new particles.
        """
        # Clear tape
        self.tape = None
        
        num_p = particles.shape[0]
        self.n_particles = num_p
        
        # 1. Normalize positions to [0.1, 0.9] to avoid boundary issues
        # We assume input is already scaled or we use a shift
        shifted_particles = particles + self.shift
        
        # 2. Initial volumes (default to grid-based)
        if volume is None:
            vol = torch.ones(num_p, device=self.device) * (1.0 / self.grid_res)**3
        else:
            vol = volume
            
        self.solver.load_initial_data_from_torch(
            shifted_particles.contiguous(),
            vol.contiguous(),
            n_grid=self.grid_res,
            grid_lim=1.0,
            device=self.warp_device
        )
        
        # Linked Torch tensors
        self.x = shifted_particles.contiguous().detach().requires_grad_(True)
        self.v = torch.zeros_like(self.x).contiguous().detach().requires_grad_(True)
        self.F = torch.eye(3, device=self.device).repeat(self.n_particles, 1, 1).contiguous().detach().requires_grad_(True)
        self.C = torch.zeros((self.n_particles, 3, 3), device=self.device).contiguous().detach().requires_grad_(True)
        
        # [FIX] Keep initial tensors as leaves
        self.x.retain_grad()
        self.v.retain_grad()
        self.F.retain_grad()
        self.C.retain_grad()
        
        # Force Warp to use these tensors (Zero-copy)
        self.solver.import_particle_x_from_torch(self.x, clone=False, device=self.warp_device)
        self.solver.import_particle_v_from_torch(self.v, clone=False, device=self.warp_device)
        self.solver.import_particle_F_from_torch(self.F, device=self.warp_device)
        self.solver.import_particle_C_from_torch(self.C, device=self.warp_device)
        
        # Set default material params
        material_params = {
            "material": "elastic",
            "E": 1e5,
            "nu": 0.3,
            "density": 1000.0,
            "g": getattr(self.cfg, 'gravity', [0.0, 0.0, -9.8]),
            "grid_v_damping_scale": getattr(self.cfg, 'grid_v_damping_scale', 1.0)
        }
        self.solver.set_parameters_dict(material_params, device=self.warp_device)
        self.solver.finalize_mu_lam(device=self.warp_device)
        
        # [NEW] Apply Boundary Conditions during reset
        self._apply_boundary()

        # Handle Controller Connections
        if controller_pos is not None and controller_pos.shape[0] > 0:
            from pytorch3d.ops import knn_points
            shifted_controller = controller_pos + self.shift
            num_ctrl = shifted_controller.shape[0]
            K = getattr(self.cfg, 'controller_max_neighbors', 16)
            
            dist, idx, _ = knn_points(
                shifted_controller.unsqueeze(0), 
                shifted_particles.unsqueeze(0), 
                K=K
            )
            
            dist = dist.squeeze(0) # [C, K]
            idx = idx.squeeze(0)   # [C, K]
            
            radius = getattr(self.cfg, 'controller_radius', 0.15)
            mask = dist.sqrt() < radius # [C, K]

            # Warp allocation
            self.controller_indices_wp = wp.from_torch(idx.to(torch.int32), dtype=wp.int32)
            self.controller_mask_wp = wp.from_torch(mask.to(torch.int32), dtype=wp.int32)
            
            # Initial offsets for PD control: offset = ctrl_initial - particle_initial
            # so kernel error = ctrl_current - particle_current - (ctrl_initial - particle_initial)
            #                  = delta_ctrl - delta_particle  (zero when particle follows controller)
            ctrl_pos_expanded = shifted_controller.unsqueeze(1).expand(-1, K, -1)
            obj_pos_expanded = shifted_particles[idx]
            initial_offsets = ctrl_pos_expanded - obj_pos_expanded
            self.initial_offsets_wp = wp.from_torch(initial_offsets.contiguous(), dtype=wp.vec3)
            
            # Particle count for normalization
            count_per_particle = torch.zeros(self.n_particles, device=self.device)
            # Count how many controller points connect to each particle
            valid_indices = idx[mask]
            if valid_indices.numel() > 0:
                count_per_particle.index_add_(0, valid_indices, torch.ones_like(valid_indices, dtype=torch.float32))
            self.count_per_particle_wp = wp.from_torch(count_per_particle, dtype=wp.float32)
            
            self.particle_dv_wp = wp.zeros(self.n_particles, dtype=wp.vec3, device=self.warp_device)
            self.num_connections = int(mask.sum().item())
            print(f"  [Simulator] Controller connected to {self.num_connections} particles (Avg {self.num_connections/num_ctrl:.1f} per ctrl point)")
        else:
            self.controller_indices_wp = None
            self.num_connections = 0

    def step(self, moe_params_wp, controller_pos=None, controller_vel=None, residual_v=None,
             stiffness_override=None, damping_override=None, kp_wp=None, kd_wp=None):
        """
        Execute one MPM step.
        
        kp_wp / kd_wp: Pre-built wp.array(dtype=float, shape=1) for differentiable
            controller gains. When provided, they are used directly (enables tape gradient flow).
        stiffness_override / damping_override: Scalar fallback (float) for inference.
        """
        _kp_wp = kp_wp
        _kd_wp = kd_wp
        if self.controller_indices_wp is not None and controller_pos is not None:
            if _kp_wp is None:
                kp_val = float(stiffness_override) if stiffness_override is not None else 1000.0
                _kp_wp = wp.from_torch(torch.tensor([kp_val], device=self.device, dtype=torch.float32),
                                       dtype=wp.float32, requires_grad=False)
            if _kd_wp is None:
                kd_val = float(damping_override) if damping_override is not None else 0.0
                _kd_wp = wp.from_torch(torch.tensor([kd_val], device=self.device, dtype=torch.float32),
                                       dtype=wp.float32, requires_grad=False)

        def _logic():
            # A. Apply Controller PD Forces
            if self.controller_indices_wp is not None and controller_pos is not None:
                shifted_ctrl_pos = controller_pos + self.shift
                v_ctrl = controller_vel if controller_vel is not None else torch.zeros_like(shifted_ctrl_pos)
                
                ctrl_pos_wp = torch2warp_vec3(shifted_ctrl_pos)
                ctrl_vel_wp = torch2warp_vec3(v_ctrl)
                clamp_val = float(getattr(self.cfg, 'controller_clamp_dv', 2.0))
                
                wp.launch(set_vec3_to_zero, self.n_particles, [self.particle_dv_wp], device=self.warp_device)
                wp.launch(accumulate_pd_forces_kernel, 
                          dim=(self.controller_indices_wp.shape[0], self.controller_indices_wp.shape[1]),
                          inputs=[self.particle_dv_wp, self.solver.mpm_state.particle_x, self.solver.mpm_state.particle_v,
                                  ctrl_pos_wp, ctrl_vel_wp, self.controller_indices_wp, self.controller_mask_wp,
                                  self.initial_offsets_wp, self.count_per_particle_wp, _kp_wp, _kd_wp, self.dt],
                          device=self.warp_device)
                wp.launch(apply_dv_with_clamping_kernel, self.n_particles, 
                          [self.solver.mpm_state.particle_v, self.particle_dv_wp, clamp_val],
                          device=self.warp_device)

            # B. Inject Neural Residual
            if residual_v is not None:
                residual_wp = torch2warp_vec3(residual_v)
                wp.launch(add_vec3_to_vec3, self.n_particles, [self.solver.mpm_state.particle_v, residual_wp], device=self.warp_device)

            # C. Solver Update (P2G2P)
            svd_clamp_max = float(getattr(self.cfg, 'svd_clamp_max', 2.0))
            self.solver.p2g2p(0, self.dt, device=self.warp_device, moe_params=moe_params_wp, svd_clamp_max=svd_clamp_max)

        if self.tape:
            with self.tape:
                _logic()
        else:
            _logic()
        
        # Export particle state to PyTorch tensors.
        # When tape is active, keep requires_grad so PyTorch backward can compute
        # gradients that are then bridged back to Warp via the sum_vec3 pattern.
        self.x = self.solver.export_particle_x_to_torch().detach().requires_grad_(True)
        self.v = self.solver.export_particle_v_to_torch().detach().requires_grad_(True)
        
        return self.x - self.shift

    def _apply_boundary(self):
        """
        Apply boundary conditions from config.
        """
        surface_type = getattr(self.cfg.boundary, 'surface', 'slip')
        friction_val = getattr(self.cfg.boundary, 'friction', 0.5) if surface_type != 'sticky' else 0.0
        
        # [FIX] Shift boundary point to match normalized grid space
        # World space [0,0,0] -> Grid space [0.5, 0.5, 0.5]
        base_point = getattr(self.cfg.boundary, 'point', [0.0, 0.0, 0.0])
        shifted_point = [base_point[0] + 0.5, base_point[1] + 0.5, base_point[2] + 0.5]
        
        self.solver.add_surface_collider(
            point=shifted_point,
            normal=getattr(self.cfg.boundary, 'normal', [0.0, 0.0, 1.0]), 
            surface=surface_type,
            friction=friction_val
        )
