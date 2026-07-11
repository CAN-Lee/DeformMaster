import torch

def apply_plasticity_return_mapping(F, mu, yield_stress, viscosity, dt, scale_ys=1.0, scale_pv=1.0):
    """
    Vectorized PyTorch implementation of Implicit Return Mapping for Viscoplasticity.
    Based on Hencky strain (log-strain) and Von-Mises yield criterion with StVK correction.
    
    Args:
        F: [N, 3, 3] Deformation gradient
        mu: [N] or [1] Shear modulus
        yield_stress: [N] or [1] Yield stress threshold
        viscosity: [N] or [1] Plastic viscosity (η)
        dt: float, time step
        scale_ys: float, scaling factor for yield stress (default 1.0)
        scale_pv: float, scaling factor for viscosity (default 1.0)
    
    Returns:
        F_new: [N, 3, 3] Updated deformation gradient
    """
    N = F.shape[0]
    device = F.device
    I = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    
    # 1. SVD with Gradient Guard
    # Add small epsilon to diagonal to prevent NaN gradients on singular values
    U, S, Vh = torch.linalg.svd(F + I * 1e-6)
    
    # 2. Stability Protection for Logarithm
    # Prevent log(0) in extreme compression
    sig = torch.clamp(S, min=0.01)
    
    # 3. Compute Hencky Strain (Log-strain)
    # epsilon = ln(sig)
    eps = torch.log(sig) # [N, 3]
    
    # 4. Deviatoric Split
    # epsilon_hat = eps - trace(eps)/3
    trace_eps = torch.sum(eps, dim=1, keepdim=True) # [N, 1]
    eps_hat = eps - trace_eps / 3.0
    
    # 5. Trial Deviatoric Stress
    # s_trial = 2 * mu * eps_hat
    mu_v = mu.view(-1, 1) if mu.dim() > 0 else mu
    s_trial = 2.0 * mu_v * eps_hat # [N, 3]
    s_trial_norm = torch.norm(s_trial, dim=1, keepdim=True) # [N, 1]
    
    # 6. Yield Condition Check
    # y = ||s_trial|| - sqrt(2/3) * yield_stress
    ys_v = yield_stress.view(-1, 1) if yield_stress.dim() > 0 else yield_stress
    y = s_trial_norm - torch.sqrt(torch.tensor(2.0/3.0, device=device)) * scale_ys * ys_v
    
    # 7. Implicit Return Mapping (only where y > 0)
    # If not yielding, F_new = F_trial
    mask = (y > 0).float()
    
    # StVK Correction for Effective Shear Modulus (mu_hat)
    # b_trial = sig^2
    b_trial = sig * sig
    mu_hat = mu_v * torch.mean(b_trial, dim=1, keepdim=True) # (sig[0]^2 + sig[1]^2 + sig[2]^2) / 3
    
    # s_new_norm = s_trial_norm - y / (1 + η / (2 * mu_hat * dt))
    visc_v = viscosity.view(-1, 1) if viscosity.dim() > 0 else viscosity
    denom = 1.0 + (scale_pv * visc_v) / (2.0 * mu_hat * dt + 1e-9)
    s_new_norm = s_trial_norm - y / denom
    
    # Reconstruct epsilon_new
    # s_new = (s_new_norm / s_trial_norm) * s_trial
    s_new = (s_new_norm / (s_trial_norm + 1e-9)) * s_trial
    
    # epsilon_new = s_new / (2 * mu) + trace(eps)/3
    eps_new = s_new / (2.0 * mu_v + 1e-9) + trace_eps / 3.0
    
    # 8. Blend between elastic and plastic states
    # Use torch.where for vectorized selection
    eps_final = torch.where(mask > 0.5, eps_new, eps)
    
    # 9. Reconstruct F
    sig_new = torch.exp(eps_final)
    F_new = torch.matmul(U, torch.matmul(torch.diag_embed(sig_new), Vh))
    
    # Final safety clamp
    return torch.clamp(F_new, min=-2.0, max=2.0)

def youngs_poisson_to_lame(E, nu):
    """
    Convert Young's Modulus (E) and Poisson's ratio (nu) to Lamé constants (mu, lambda).
    
    Args:
        E: Young's Modulus
        nu: Poisson's ratio
        
    Returns:
        mu: Shear modulus
        lam: First Lamé parameter
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu + 1e-6))
    return mu, lam
