import torch
from torch import Tensor
from typing import List, Tuple

def get_bspline_weights(x: Tensor, dx: float, grid_res: List[int]) -> Tuple[Tensor, Tensor]:
    """
    Compute Cubic B-Spline weights and node indices for particles.
    Includes boundary protection to prevent NaN and out-of-bounds indices.
    """
    bsz, num_particles, _ = x.shape
    device = x.device
    
    # 1. Calculate relative position to the grid, clamped to safe region [1.0, res-2.0]
    # This ensures that the 3x3x3 neighborhood always stays within [0, res-1]
    pos_grid = x / dx
    
    # Safety margin: avoid the very last node to allow 3x3x3 window
    # grid_res is [rx, ry, rz], convert to float tensor for broadcasting
    res_tensor = torch.tensor(grid_res, device=device).float()
    limit = res_tensor - 2.1
    
    # Use broadcastable tensors for clamp
    pos_grid = torch.max(pos_grid, torch.tensor(1.1, device=device))
    pos_grid = torch.min(pos_grid, limit.view(1, 1, 3))
    
    base_idx = (pos_grid - 0.5).floor().long() 
    fx = pos_grid - base_idx.float() # distance to base node (0.5 ~ 1.5)
    
    # 2. Cubic B-Spline kernels
    w = [
        0.5 * (1.5 - fx)**2,
        0.75 - (fx - 1.0)**2,
        0.5 * (fx - 0.5)**2
    ] 
    
    # 3. Compute 3D weights and indices
    weights_3d = []
    node_idxs_3d = []
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w_ijk = w[i][..., 0] * w[j][..., 1] * w[k][..., 2]
                weights_3d.append(w_ijk)
                
                idx_ijk = base_idx + torch.tensor([i, j, k], device=device).view(1, 1, 3)
                node_idxs_3d.append(idx_ijk)
                
    weights = torch.stack(weights_3d, dim=-1)
    node_idxs = torch.stack(node_idxs_3d, dim=-2)
    
    return weights, node_idxs

def get_grid_locations(x: Tensor, grid_res: List[int], dx: float) -> Tuple[Tensor, Tensor]:
    """
    Legacy grid center mapping (for comparison or initial simple tests).
    """
    grid_idxs = torch.floor(x / dx).to(torch.long)
    grid_idxs[..., 0] = torch.clamp(grid_idxs[..., 0], 0, grid_res[0] - 1)
    grid_idxs[..., 1] = torch.clamp(grid_idxs[..., 1], 0, grid_res[1] - 1)
    grid_idxs[..., 2] = torch.clamp(grid_idxs[..., 2], 0, grid_res[2] - 1)
    grid_pos = (grid_idxs.float() + 0.5) * dx
    return grid_pos, grid_idxs

def positional_encoding(tensor, num_encoding_functions=6):
    if num_encoding_functions == 0:
        return tensor

    encoding = [tensor]
    frequency_bands = 2.0 ** torch.linspace(
        0.0,
        num_encoding_functions - 1,
        num_encoding_functions,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    return torch.cat(encoding, dim=-1)
