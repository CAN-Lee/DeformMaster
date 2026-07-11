import warp as wp
import math

# DeformMaster MoE Stress Models implemented in Warp
# All functions return Kirchhoff Stress (tau = P F^T)

@wp.func
def safe_determinant(F: wp.mat33):
    det = wp.determinant(F)
    return wp.clamp(det, 0.01, 10.0)

@wp.func
def safe_inverse_transpose(F: wp.mat33):
    # Standard Warp inverse might be unstable for singular matrices
    # But for now we use it with safe_determinant logic if possible
    # Or just use the formula for 3x3 inverse
    det = wp.determinant(F)
    if wp.abs(det) < 1e-6:
        # Add perturbation
        F_safe = F + wp.mat33(1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6)
        return wp.transpose(wp.inverse(F_safe))
    return wp.transpose(wp.inverse(F))

@wp.func
def safe_F(F: wp.mat33, clamp_max: float):
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    S = wp.vec3(0.0)
    # Asymmetric epsilon to break singular value degeneracy for stable SVD backward
    F_stab = F + wp.mat33(1e-4, 0.0, 0.0, 0.0, 2e-4, 0.0, 0.0, 0.0, 3e-4)
    wp.svd3(F_stab, U, S, V)
    
    S_clamped = wp.vec3(
        wp.clamp(S[0], 0.05, clamp_max),
        wp.clamp(S[1], 0.05, clamp_max),
        wp.clamp(S[2], 0.05, clamp_max)
    )
    
    # Reconstruct F
    # diag(S)
    DS = wp.mat33(
        S_clamped[0], 0.0, 0.0,
        0.0, S_clamped[1], 0.0,
        0.0, 0.0, S_clamped[2]
    )
    return U * DS * wp.transpose(V)

@wp.func
def compute_tau_nh(F: wp.mat33, mu: float, lam: float, clamp_max: float):
    F_c = safe_F(F, clamp_max)
    J = safe_determinant(F_c)
    lnJ = wp.log(J)
    b = F_c * wp.transpose(F_c)
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return mu * (b - I) + lam * lnJ * I

@wp.func
def compute_tau_co(F: wp.mat33, mu: float, lam: float, clamp_max: float):
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    S = wp.vec3(0.0)
    F_stab = F + wp.mat33(1e-4, 0.0, 0.0, 0.0, 2e-4, 0.0, 0.0, 0.0, 3e-4)
    wp.svd3(F_stab, U, S, V)
    R = U * wp.transpose(V)
    
    F_c = safe_F(F, clamp_max)
    J = safe_determinant(F_c)
    b = F_c * wp.transpose(F_c)
    RFt = R * wp.transpose(F_c)
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (b - RFt) + lam * (J - 1.0) * J * I

@wp.func
def compute_tau_st(F: wp.mat33, mu: float, lam: float, clamp_max: float):
    F_c = safe_F(F, clamp_max)
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    FtF = wp.transpose(F_c) * F_c
    E = 0.5 * (FtF - I)
    trE = E[0,0] + E[1,1] + E[2,2]
    S = 2.0 * mu * E + lam * trE * I
    return F_c * S * wp.transpose(F_c)

@wp.func
def compute_tau_fi(F: wp.mat33, k: float, d: wp.vec3, clamp_max: float):
    F_c = safe_F(F, clamp_max)
    Fd = F_c * d
    l = wp.length(Fd)
    if l < 1.0:
        return wp.mat33(0.0)
    
    ddT = wp.mat33(
        d[0]*d[0], d[0]*d[1], d[0]*d[2],
        d[1]*d[0], d[1]*d[1], d[1]*d[2],
        d[2]*d[0], d[2]*d[1], d[2]*d[2]
    )
    
    scalar = k * (l - 1.0) / (l + 1e-6)
    return scalar * F_c * ddT * wp.transpose(F_c)

@wp.kernel
def compute_moe_stress_kernel(
    particle_F: wp.array(dtype=wp.mat33),
    particle_weights: wp.array(dtype=float, ndim=2), # [N, 4]
    particle_mu: wp.array(dtype=float),
    particle_lam: wp.array(dtype=float),
    particle_fk: wp.array(dtype=float),
    particle_fdir: wp.array(dtype=wp.vec3),
    particle_stress: wp.array(dtype=wp.mat33), # Output Kirchhoff Stress
    active_mask: wp.array(dtype=int, ndim=1), # [4] 1 if expert is active
    svd_clamp_max: float
):
    p = wp.tid()
    F = particle_F[p]
    mu = particle_mu[p]
    lam = particle_lam[p]
    fk = particle_fk[p]
    fdir = particle_fdir[p]
    
    tau_total = wp.mat33(0.0)
    
    if active_mask[0] == 1:
        tau_total = tau_total + particle_weights[p, 0] * compute_tau_nh(F, mu, lam, svd_clamp_max)
    if active_mask[1] == 1:
        tau_total = tau_total + particle_weights[p, 1] * compute_tau_co(F, mu, lam, svd_clamp_max)
    if active_mask[2] == 1:
        tau_total = tau_total + particle_weights[p, 2] * compute_tau_st(F, mu, lam, svd_clamp_max)
    if active_mask[3] == 1:
        tau_total = tau_total + particle_weights[p, 3] * compute_tau_fi(F, fk, fdir, svd_clamp_max)
        
    particle_stress[p] = 0.5 * (tau_total + wp.transpose(tau_total))
