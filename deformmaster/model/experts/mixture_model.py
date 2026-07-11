
import torch
import torch.nn as nn
from ...utils.mpm_utils import apply_plasticity_return_mapping

class HenckyViscoPlasticity(nn.Module):
    """
    Standard Hencky-strain based Viscoplasticity (Return Mapping).
    Inheriting from Plasticity allows it to be used as a formal state correction module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, F, mu, yield_stress, viscosity, dt):
        """
        Args:
            F: [N, 3, 3] Deformation gradient
            mu: [N] Shear modulus
            yield_stress: [N] Yield stress threshold
            viscosity: [N] Plastic viscosity
            dt: float, time step
        """
        return apply_plasticity_return_mapping(F, mu, yield_stress, viscosity, dt)

class NeoHookeanElasticity(nn.Module):
    """
    Standard Neo-Hookean Model (Expert A)
    Psi = 0.5 * mu * (I1 - 3) - mu * ln(J) + 0.5 * lam * (ln(J))^2
    """
    def __init__(self, svd_clamp_max=2.0):
        super().__init__()
        self.svd_clamp_max = svd_clamp_max

    def _safe_F(self, F):
        # [STABILITY] SVD clamping to prevent numerical explosions
        U, S, Vh = torch.linalg.svd(F + torch.eye(3, device=F.device, dtype=F.dtype) * 1e-5)
        S_clamped = torch.clamp(S, min=0.05, max=self.svd_clamp_max)
        return torch.matmul(U, torch.matmul(torch.diag_embed(S_clamped), Vh))

    def forward(self, F: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        # F: [B, N, 3, 3] or [N, 3, 3]
        I = torch.eye(3, device=F.device, dtype=F.dtype)
        
        # [STABILITY] Use SVD clamping instead of element-wise clamping
        F_clamped = self._safe_F(F)
        
        # [STABILITY] Add a diag perturbation
        F_stab = F_clamped + I * 1e-5
        
        # [STABILITY] Use pseudo-inverse
        FinvT = torch.linalg.pinv(F_stab).transpose(-2, -1)
        
        # [STABILITY] Robust determinant - relaxed for cloth compression
        detF = torch.det(F_stab)
        J = torch.clamp(detF, min=0.01, max=10.0).unsqueeze(-1).unsqueeze(-1) # [..., 1, 1]
        
        # P = mu * (F - F^-T) + lam * ln(J) * F^-T
        lnJ = torch.log(J)
        
        term1 = mu.view(-1, 1, 1) * (F_clamped - FinvT)
        term2 = lam.view(-1, 1, 1) * lnJ * FinvT
        
        P = term1 + term2
        
        # [STABILITY] Final Stress Clipping - tightened to 1e5 for cloth
        P = torch.clamp(P, min=-1e5, max=1e5)
        return P

class FiberElasticity(nn.Module):
    """
    Anisotropic Fiber Model (Expert D)
    Psi = 0.5 * k * (|Fd| - 1)^2 (only if extended)
    """
    def __init__(self, svd_clamp_max=2.0):
        super().__init__()
        self.svd_clamp_max = svd_clamp_max

    def _safe_F(self, F):
        # [STABILITY] SVD clamping
        U, S, Vh = torch.linalg.svd(F + torch.eye(3, device=F.device, dtype=F.dtype) * 1e-5)
        S_clamped = torch.clamp(S, min=0.05, max=self.svd_clamp_max)
        return torch.matmul(U, torch.matmul(torch.diag_embed(S_clamped), Vh))

    def forward(self, F: torch.Tensor, k: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # F: [N, 3, 3]
        # k: [N] stiffness
        # d: [N, 3] fiber direction (normalized)
        
        # [STABILITY] Use SVD clamping
        F_clamped = self._safe_F(F)
        
        # Fd = F * d
        # [STABILITY] Ensure d is not NaN and is normalized
        d_norm = torch.nn.functional.normalize(d, dim=-1, eps=1e-8)
        d_vec = d_norm.unsqueeze(-1) # [N, 3, 1]
        Fd = torch.matmul(F_clamped, d_vec) # [N, 3, 1]
        
        l = torch.norm(Fd, dim=1, keepdim=True) # [N, 1, 1]
        
        # Only penalize extension (l > 1)
        mask = (l > 1.0).float()
        
        # P = k * (l - 1)/l * F * (d \otimes d)
        # d \otimes d -> [N, 3, 3]
        ddT = torch.matmul(d_vec, d_vec.transpose(-2, -1))
        
        # [STABILITY] Avoid division by zero
        scalar = k.view(-1, 1, 1) * (l - 1.0) / (l + 1e-6) * mask
        
        P = scalar * torch.matmul(F_clamped, ddT)
        return torch.clamp(P, min=-2e5, max=2e5) # [STABILITY] Clamp fiber stress

class MixtureElasticity(nn.Module):
    """
    The Master Constitutive Model that blends experts based on configuration.
    """
    def __init__(self, active_experts=['nh', 'co', 'st', 'fi'], svd_clamp_max=2.0):
        super().__init__()
        self.active_experts = active_experts
        self.expert_nh = NeoHookeanElasticity(svd_clamp_max=svd_clamp_max)
        self.expert_fiber = FiberElasticity(svd_clamp_max=svd_clamp_max)
        
        # Map expert keys to their calculation functions
        self.expert_map = {
            'nh': self._compute_nh,
            'co': self._compute_co,
            'st': self._compute_st,
            'fi': self._compute_fi
        }

    def _compute_nh(self, F, mu, lam, fiber_k, fiber_dir):
        return self.expert_nh(F, mu, lam)
    
    def _compute_co(self, F, mu, lam, fiber_k, fiber_dir):
        return self._manual_corotated(F, mu, lam)
    
    def _compute_st(self, F, mu, lam, fiber_k, fiber_dir):
        return self._manual_stvk(F, mu, lam)
    
    def _compute_fi(self, F, mu, lam, fiber_k, fiber_dir):
        return self.expert_fiber(F, fiber_k, fiber_dir)

    def forward(self, F: torch.Tensor, log_E=None, nu=None) -> torch.Tensor:
        if not hasattr(self, 'current_params'):
            raise RuntimeError("MPM params not set! Set model.current_params before simulation step.")
            
        params = self.current_params
        return self._compute_mixture_stress(F, params)

    def _compute_mixture_stress(self, F, params):
        weights = params['weights'] # [N, num_active]
        mu = params['mu']
        lam = params['lam']
        fk = params['fiber_k']
        fd = params['fiber_dir']
        
        P_total = torch.zeros_like(F)
        
        for i, expert_key in enumerate(self.active_experts):
            if expert_key in self.expert_map:
                w = weights[..., i].view(-1, 1, 1)
                P_expert = self.expert_map[expert_key](F, mu, lam, fk, fd)
                P_total = P_total + w * P_expert
                
        return torch.clamp(P_total, min=-2e5, max=2e5) # [STABILITY] Final mixture stress clipping

    def _manual_corotated(self, F, mu, lam):
        # Re-implementation to accept mu, lam directly
        # [STABILITY] Add epsilon before SVD to prevent NaN gradients on identical singular values
        I = torch.eye(3, device=F.device).to(F.dtype)
        F_safe = F + I * 1e-6
        
        # [STABILITY] Use SVD with a gradient-safe implementation if possible, 
        # or at least ensure we don't crash on singular matrices.
        try:
            U, sigma, Vh = torch.linalg.svd(F_safe)
        except RuntimeError:
            # Fallback for SVD failure
            U, sigma, Vh = torch.linalg.svd(F_safe + torch.randn_like(F_safe) * 1e-6)
            
        R = torch.matmul(U, Vh)
        
        # [STABILITY] Robust determinant and inverse
        J = torch.clamp(torch.det(F_safe), min=0.01, max=10.0).view(-1, 1, 1)
        FinvT = torch.linalg.pinv(F_safe).transpose(-2, -1)
        
        term1 = 2.0 * mu.view(-1, 1, 1) * (F - R)
        term2 = lam.view(-1, 1, 1) * (J - 1.0) * J * FinvT
        
        P = term1 + term2
        return torch.clamp(P, min=-2e5, max=2e5)

    def _manual_stvk(self, F, mu, lam):
        I = torch.eye(3, device=F.device, dtype=F.dtype)
        # [STABILITY] Clamp F to prevent extreme values in F^T * F
        F_clamped = torch.clamp(F, min=-5.0, max=5.0) # Tightened from 10.0
        E = 0.5 * (torch.matmul(F_clamped.transpose(-2, -1), F_clamped) - I)
        trE = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        
        S = 2.0 * mu.view(-1, 1, 1) * E + lam.view(-1, 1, 1) * trE * I
        P = torch.matmul(F_clamped, S)
        return torch.clamp(P, min=-2e5, max=2e5)
