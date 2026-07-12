"""
PhysFlow-style particle filling using Gaussian reconstruction from gaussian_output.

Uses density field from Gaussians (pos, opacity, cov) -> fill_dense_grids -> internal_filling
(ray-cast inside/outside). Optional Taichi backend for speed; fallback pure NumPy.
"""

import os
import numpy as np
import torch

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

# Optional Taichi backend (parallel kernels)
TI_AVAILABLE = False
try:
    import taichi as ti
    ti.init(arch=ti.cpu)  # use CPU for compatibility; set ti.gpu if available
    TI_AVAILABLE = True
except Exception:
    ti = None

# ---------------------------------------------------------------------------
# Taichi kernels (when TI_AVAILABLE): assign, densify, fill_dense, internal
# All buffers are ti.template() (fields); data copied to/from numpy in caller.
# ---------------------------------------------------------------------------
if TI_AVAILABLE:
    @ti.func
    def _ti_cov_inv_from_6(c6: ti.types.vector(6, float)) -> ti.types.matrix(3, 3, float):
        c = ti.Matrix([
            [c6[0], c6[1], c6[2]],
            [c6[1], c6[3], c6[4]],
            [c6[2], c6[4], c6[5]],
        ])
        c = c + 1e-10 * ti.Matrix.identity(float, 3)
        return c.inverse()

    @ti.kernel
    def _ti_assign_pos_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
        grid_n = grid.shape[0]
        for pi in range(pos.shape[0]):
            p = pos[pi]
            i = ti.cast(ti.floor(p[0] / grid_dx), int)
            j = ti.cast(ti.floor(p[1] / grid_dx), int)
            k = ti.cast(ti.floor(p[2] / grid_dx), int)
            i = ti.max(0, ti.min(i, grid_n - 1))
            j = ti.max(0, ti.min(j, grid_n - 1))
            k = ti.max(0, ti.min(k, grid_n - 1))
            ti.atomic_add(grid[i, j, k], 1)

    @ti.kernel
    def _ti_densify_grids(
        pos: ti.template(),
        opacity: ti.template(),
        cov: ti.template(),
        grid_density: ti.template(),
        grid_dx: float,
    ):
        grid_n = grid_density.shape[0]
        r = 5
        for pi in range(pos.shape[0]):
            p = pos[pi]
            i0 = ti.cast(ti.floor(p[0] / grid_dx), int)
            j0 = ti.cast(ti.floor(p[1] / grid_dx), int)
            k0 = ti.cast(ti.floor(p[2] / grid_dx), int)
            c6 = ti.Vector([cov[pi, 0], cov[pi, 1], cov[pi, 2], cov[pi, 3], cov[pi, 4], cov[pi, 5]])
            cov_inv = _ti_cov_inv_from_6(c6)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        i = i0 + di
                        j = j0 + dj
                        k = k0 + dk
                        if 0 <= i < grid_n and 0 <= j < grid_n and 0 <= k < grid_n:
                            cell_center = (ti.Vector([float(i), float(j), float(k)]) + 0.5) * grid_dx
                            dist = cell_center - p
                            quad = dist.dot(cov_inv @ dist)
                            w = opacity[pi] * ti.exp(-0.5 * quad)
                            ti.atomic_add(grid_density[i, j, k], w)

    @ti.kernel
    def _ti_fill_dense(
        grid: ti.template(),
        grid_density: ti.template(),
        grid_dx: float,
        density_thres: float,
        new_particles: ti.template(),
        counter: ti.template(),
        max_ppc: int,
        max_samples: int,
        ox: float, oy: float, oz: float,
    ):
        grid_n = grid.shape[0]
        for i in range(grid_n):
            for j in range(grid_n):
                for k in range(grid_n):
                    if grid_density[i, j, k] > density_thres:
                        cur = grid[i, j, k]
                        if cur < max_ppc:
                            diff = max_ppc - cur
                            start = ti.atomic_add(counter[0], diff)
                            actual = diff
                            if start + diff > max_samples:
                                actual = ti.max(0, max_samples - start)
                                ti.atomic_sub(counter[0], diff - actual)
                            if actual > 0:
                                grid[i, j, k] = cur + actual
                                for d in range(actual):
                                    idx = start + d
                                    new_particles[idx][0] = (i + ti.random(float)) * grid_dx + ox
                                    new_particles[idx][1] = (j + ti.random(float)) * grid_dx + oy
                                    new_particles[idx][2] = (k + ti.random(float)) * grid_dx + oz

    @ti.func
    def _ti_collision_search(grid_density: ti.template(), i: int, j: int, k: int, dir_type: int, grid_n: int, threshold: float) -> bool:
        di, dj, dk = 0, 0, 0
        if dir_type == 0: di = 1
        elif dir_type == 1: di = -1
        elif dir_type == 2: dj = 1
        elif dir_type == 3: dj = -1
        elif dir_type == 4: dk = 1
        elif dir_type == 5: dk = -1
        ni, nj, nk = i + di, j + dj, k + dk
        found = False
        while ni >= 0 and ni < grid_n and nj >= 0 and nj < grid_n and nk >= 0 and nk < grid_n:
            if grid_density[ni, nj, nk] > threshold:
                found = True
                break
            ni += di
            nj += dj
            nk += dk
        return found

    @ti.func
    def _ti_collision_times(grid_density: ti.template(), i: int, j: int, k: int, dir_type: int, grid_n: int, threshold: float) -> int:
        di, dj, dk = 0, 0, 0
        if dir_type == 0: di = 1
        elif dir_type == 1: di = -1
        elif dir_type == 2: dj = 1
        elif dir_type == 3: dj = -1
        elif dir_type == 4: dk = 1
        elif dir_type == 5: dk = -1
        state = grid_density[i, j, k] > threshold
        ni, nj, nk = i + di, j + dj, k + dk
        times = 0
        while ni >= 0 and ni < grid_n and nj >= 0 and nj < grid_n and nk >= 0 and nk < grid_n:
            new_state = grid_density[ni, nj, nk] > threshold
            if not state and new_state:
                times += 1
            state = new_state
            ni += di
            nj += dj
            nk += dk
        return times

    @ti.kernel
    def _ti_internal_filling(
        grid: ti.template(),
        grid_density: ti.template(),
        grid_dx: float,
        new_particles: ti.template(),
        counter: ti.template(),
        max_ppc: int,
        max_samples: int,
        exclude_dir: int,
        ray_cast_dir: int,
        threshold: float,
        ox: float, oy: float, oz: float,
    ):
        grid_n = grid.shape[0]
        for i in range(grid_n):
            for j in range(grid_n):
                for k in range(grid_n):
                    if grid[i, j, k] == 0:
                        hit_all = True
                        for d in ti.static(range(6)):
                            if d != exclude_dir:
                                if not _ti_collision_search(grid_density, i, j, k, d, grid_n, threshold):
                                    hit_all = False
                        if hit_all:
                            times = _ti_collision_times(grid_density, i, j, k, ray_cast_dir, grid_n, threshold)
                            if times % 2 == 1:
                                diff = max_ppc
                                start = ti.atomic_add(counter[0], diff)
                                actual = diff
                                if start + diff > max_samples:
                                    actual = ti.max(0, max_samples - start)
                                    ti.atomic_sub(counter[0], diff - actual)
                                if actual > 0:
                                    grid[i, j, k] = actual
                                    for d in range(actual):
                                        idx = start + d
                                        new_particles[idx][0] = (i + ti.random(float)) * grid_dx + ox
                                        new_particles[idx][1] = (j + ti.random(float)) * grid_dx + oy
                                        new_particles[idx][2] = (k + ti.random(float)) * grid_dx + oz


# ---------- Gaussian covariance from PLY (scale + rotation) ----------
def _build_rotation_quat(r):
    """r: (N, 4) quaternion [w, x, y, z]. Returns (N, 3, 3)."""
    norm = np.sqrt((r ** 2).sum(axis=1, keepdims=True))
    q = r / np.clip(norm, 1e-8, None)
    R = np.zeros((len(q), 3, 3), dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_scaling_rotation(s, r):
    """s: (N, 3) scale (will be exp(s)), r: (N, 4) quat. Returns (N, 3, 3) L = R @ diag(s)."""
    scale = np.exp(s)
    R = _build_rotation_quat(r)
    L = np.zeros_like(R)
    L[:, 0, 0] = scale[:, 0]
    L[:, 1, 1] = scale[:, 1]
    L[:, 2, 2] = scale[:, 2]
    L = np.einsum("nij,njk->nik", R, L)
    return L


def _strip_symmetric(sym):
    """sym: (N, 3, 3). Return (N, 6) lower diag [00, 01, 02, 11, 12, 22]."""
    out = np.zeros((sym.shape[0], 6), dtype=sym.dtype)
    out[:, 0] = sym[:, 0, 0]
    out[:, 1] = sym[:, 0, 1]
    out[:, 2] = sym[:, 0, 2]
    out[:, 3] = sym[:, 1, 1]
    out[:, 4] = sym[:, 1, 2]
    out[:, 5] = sym[:, 2, 2]
    return out


def load_gaussian_ply(ply_path):
    """
    Load 3DGS PLY and return pos (N,3), opacity (N,), cov (N,6) for filling.
    Opacity is sigmoid(ply_opacity). Covariance from scale_* and rot_* (exp(scale), quat -> L, cov = L@L.T).
    """
    if PlyData is None:
        raise ImportError("plyfile is required: pip install plyfile")
    ply = PlyData.read(ply_path)
    vert = ply["vertex"]
    xyz = np.stack([np.asarray(vert["x"]), np.asarray(vert["y"]), np.asarray(vert["z"])], axis=1)
    raw_opacity = np.asarray(vert["opacity"])
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))

    names = vert.data.dtype.names
    scale_names = sorted([p for p in names if p.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
    rot_names = sorted([p for p in names if p.startswith("rot")], key=lambda x: int(x.split("_")[-1]))
    scales = np.stack([np.asarray(vert[n]) for n in scale_names], axis=1)
    # Isotropic PLY (e.g. iso=True) has only scale_0: replicate to (N, 3)
    if scales.shape[1] == 1:
        scales = np.repeat(scales, 3, axis=1)
    rots = np.stack([np.asarray(vert[n]) for n in rot_names], axis=1)
    L = _build_scaling_rotation(scales, rots)
    cov_3x3 = np.einsum("nij,nkj->nik", L, L)
    cov = _strip_symmetric(cov_3x3)
    return xyz.astype(np.float32), opacity.astype(np.float32), cov.astype(np.float32)


# ---------- Density field (Gaussian splat onto grid) ----------
def _gaussian_weight(cell_center, pos, cov_upper, opacity, grid_dx):
    """Single Gaussian contribution at one cell. cov_upper: (6,) upper triangle."""
    dist = cell_center - pos
    cov = np.array([
        [cov_upper[0], cov_upper[1], cov_upper[2]],
        [cov_upper[1], cov_upper[3], cov_upper[4]],
        [cov_upper[2], cov_upper[4], cov_upper[5]],
    ], dtype=np.float64)
    try:
        cov_inv = np.linalg.inv(cov + 1e-10 * np.eye(3))
    except np.linalg.LinAlgError:
        return 0.0
    d = dist.astype(np.float64)
    q = float(d @ cov_inv @ d)
    w = opacity * np.exp(-0.5 * q)
    return w


def densify_grids_numpy(pos, opacity, cov, grid_dx, grid_origin, grid_n):
    """
    Build grid_density (grid_n^3) from Gaussians. Each cell gets sum of opacity * exp(-0.5 * d^T cov^{-1} d).
    Only sum over Gaussians whose support (from cov eigenvalues) overlaps the cell to keep cost O(N*G) reasonable.
    """
    grid_density = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
    pos_shifted = pos - grid_origin
    for g in range(len(pos)):
        p = pos_shifted[g]
        i0, j0, k0 = (p / grid_dx).astype(int)
        cov6 = cov[g]
        cov3 = np.array([
            [cov6[0], cov6[1], cov6[2]],
            [cov6[1], cov6[3], cov6[4]],
            [cov6[2], cov6[4], cov6[5]],
        ], dtype=np.float64)
        try:
            eigs = np.linalg.eigvalsh(cov3)
            eigs = np.maximum(eigs, 1e-12)
            r = int(np.ceil(np.sqrt(np.max(eigs)) / grid_dx)) + 1
        except Exception:
            r = 3
        r = min(r, grid_n)
        i_min = max(0, i0 - r)
        i_max = min(grid_n, i0 + r + 1)
        j_min = max(0, j0 - r)
        j_max = min(grid_n, j0 + r + 1)
        k_min = max(0, k0 - r)
        k_max = min(grid_n, k0 + r + 1)
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    cell_center = (np.array([i, j, k], dtype=np.float64) + 0.5) * grid_dx
                    w = _gaussian_weight(cell_center, p.astype(np.float64), cov6.astype(np.float64), opacity[g], grid_dx)
                    grid_density[i, j, k] += w
    return grid_density


def densify_grids_batched(pos, opacity, cov, grid_dx, grid_origin, grid_n):
    """Vectorized density: for each (j,k) slice, compute density for all i (memory O(grid_n * G))."""
    grid_density = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
    pos_shifted = pos - grid_origin  # (G, 3)
    cov_inv_list = []
    for g in range(len(pos)):
        cov6 = cov[g]
        cov3 = np.array([
            [cov6[0], cov6[1], cov6[2]],
            [cov6[1], cov6[3], cov6[4]],
            [cov6[2], cov6[4], cov6[5]],
        ], dtype=np.float64)
        try:
            cov_inv_list.append(np.linalg.inv(cov3 + 1e-10 * np.eye(3)))
        except np.linalg.LinAlgError:
            cov_inv_list.append(np.eye(3) * 1e-6)
    cov_inv = np.stack(cov_inv_list, axis=0)  # (G, 3, 3)
    for k in range(grid_n):
        for j in range(grid_n):
            cell_centers = np.zeros((grid_n, 3), dtype=np.float64)
            cell_centers[:, 0] = (np.arange(grid_n) + 0.5) * grid_dx
            cell_centers[:, 1] = (j + 0.5) * grid_dx
            cell_centers[:, 2] = (k + 0.5) * grid_dx
            dist = cell_centers[:, None, :] - pos_shifted[None, :, :]
            quad = np.einsum("igk,gkl,igl->ig", dist, cov_inv, dist)
            density = (opacity[None, :] * np.exp(-0.5 * quad)).sum(axis=1)
            grid_density[:, j, k] = density.astype(np.float32)
    return grid_density


# ---------- Fill dense grids ----------
def fill_dense_grids_numpy(grid, grid_density, grid_dx, grid_origin, density_thres, max_particles_per_cell, rng=None):
    """Where grid_density > density_thres, add up to max_particles_per_cell particles per cell. Returns new particle list."""
    if rng is None:
        rng = np.random.default_rng()
    new_particles = []
    for i in range(grid_density.shape[0]):
        for j in range(grid_density.shape[1]):
            for k in range(grid_density.shape[2]):
                if grid_density[i, j, k] <= density_thres:
                    continue
                current = int(grid[i, j, k])
                add = max(0, max_particles_per_cell - current)
                for _ in range(add):
                    di, dj, dk = rng.uniform(0, 1, size=3)
                    pt = (np.array([i + di, j + dj, k + dk], dtype=np.float64) * grid_dx) + grid_origin
                    new_particles.append(pt)
                grid[i, j, k] = current + add
    return np.array(new_particles, dtype=np.float32) if new_particles else np.zeros((0, 3), dtype=np.float32)


# ---------- Internal filling: ray-cast ----------
def _collision_search(grid_density, i, j, k, dir_type, grid_n, threshold):
    """Along one direction, find if we ever hit a cell with density > threshold."""
    di = dj = dk = 0
    if dir_type == 0: di = 1
    elif dir_type == 1: di = -1
    elif dir_type == 2: dj = 1
    elif dir_type == 3: dj = -1
    elif dir_type == 4: dk = 1
    elif dir_type == 5: dk = -1
    ni, nj, nk = i + di, j + dj, k + dk
    while 0 <= ni < grid_n and 0 <= nj < grid_n and 0 <= nk < grid_n:
        if grid_density[ni, nj, nk] > threshold:
            return True
        ni += di
        nj += dj
        nk += dk
    return False


def _collision_times(grid_density, i, j, k, dir_type, grid_n, threshold):
    """Count shell crossings along one direction (for parity: odd = inside)."""
    di = dj = dk = 0
    if dir_type == 0: di = 1
    elif dir_type == 1: di = -1
    elif dir_type == 2: dj = 1
    elif dir_type == 3: dj = -1
    elif dir_type == 4: dk = 1
    elif dir_type == 5: dk = -1
    state = grid_density[i, j, k] > threshold
    ni, nj, nk = i + di, j + dj, k + dk
    times = 0
    while 0 <= ni < grid_n and 0 <= nj < grid_n and 0 <= nk < grid_n:
        new_state = grid_density[ni, nj, nk] > threshold
        if not state and new_state:
            times += 1
        state = new_state
        ni += di
        nj += dj
        nk += dk
    return times


def internal_filling_numpy(grid, grid_density, grid_dx, grid_origin, threshold, max_particles_per_cell, exclude_dir, ray_cast_dir, rng=None):
    """Fill empty cells that are inside: 6-direction hit + odd ray crossings."""
    if rng is None:
        rng = np.random.default_rng()
    grid_n = grid_density.shape[0]
    new_particles = []
    for i in range(grid_n):
        for j in range(grid_n):
            for k in range(grid_n):
                if grid[i, j, k] > 0:
                    continue
                hit_all = True
                for d in range(6):
                    if d == exclude_dir:
                        continue
                    if not _collision_search(grid_density, i, j, k, d, grid_n, threshold):
                        hit_all = False
                        break
                if not hit_all:
                    continue
                times = _collision_times(grid_density, i, j, k, ray_cast_dir, grid_n, threshold)
                if times % 2 != 1:
                    continue
                add = max_particles_per_cell
                for _ in range(add):
                    di, dj, dk = rng.uniform(0, 1, size=3)
                    pt = (np.array([i + di, j + dj, k + dk], dtype=np.float64) * grid_dx) + grid_origin
                    new_particles.append(pt)
                grid[i, j, k] = add
    return np.array(new_particles, dtype=np.float32) if new_particles else np.zeros((0, 3), dtype=np.float32)


# ---------- Assign original positions to grid (count per cell) ----------
def assign_pos_to_grid(pos, grid_origin, grid_dx, grid_n):
    grid = np.zeros((grid_n, grid_n, grid_n), dtype=np.int32)
    pos_shifted = pos - grid_origin
    idx = (pos_shifted / grid_dx).astype(int)
    idx = np.clip(idx, 0, grid_n - 1)
    for n in range(len(idx)):
        i, j, k = idx[n, 0], idx[n, 1], idx[n, 2]
        grid[i, j, k] += 1
    return grid


def _fill_particles_gaussian_taichi(
    pos_shifted,
    opacity,
    cov,
    grid_dx,
    grid_origin,
    grid_n,
    max_samples,
    density_thres,
    search_thres,
    max_particles_per_cell,
    search_exclude_dir,
    ray_cast_dir,
    seed,
):
    """Taichi-accelerated path. pos_shifted = pos - grid_origin (in grid space). Returns (new_dense, new_internal) as numpy (N,3) float32."""
    if not TI_AVAILABLE:
        raise RuntimeError("Taichi not available")
    n_pos = pos_shifted.shape[0]
    ox, oy, oz = float(grid_origin[0]), float(grid_origin[1]), float(grid_origin[2])

    ti_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_pos)
    ti_opacity = ti.field(dtype=ti.f32, shape=n_pos)
    ti_cov = ti.field(dtype=ti.f32, shape=(n_pos, 6))
    ti_pos.from_numpy(np.asarray(pos_shifted, dtype=np.float32))
    ti_opacity.from_numpy(np.asarray(opacity, dtype=np.float32))
    ti_cov.from_numpy(np.asarray(cov, dtype=np.float32).reshape(n_pos, 6))

    grid = ti.field(dtype=ti.i32, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=ti.f32, shape=(grid_n, grid_n, grid_n))
    grid.fill(0)
    grid_density.fill(0.0)

    _ti_assign_pos_to_grid(ti_pos, grid, grid_dx)
    _ti_densify_grids(ti_pos, ti_opacity, ti_cov, grid_density, grid_dx)

    new_particles = ti.Vector.field(3, dtype=ti.f32, shape=max_samples)
    counter = ti.field(dtype=ti.i32, shape=(1,))
    counter[0] = 0

    if seed is not None and hasattr(ti, "random_seed"):
        ti.random_seed(seed)
    # Taichi 1.7+ has no runtime random_seed; use ti.init(random_seed=...) for reproducibility

    _ti_fill_dense(
        grid, grid_density, grid_dx, density_thres,
        new_particles, counter, max_particles_per_cell, max_samples,
        ox, oy, oz,
    )
    n_dense = counter[0]
    counter[0] = n_dense

    _ti_internal_filling(
        grid, grid_density, grid_dx, new_particles, counter,
        max_particles_per_cell, max_samples, search_exclude_dir, ray_cast_dir, search_thres,
        ox, oy, oz,
    )
    n_internal = counter[0] - n_dense

    out = np.asarray(new_particles.to_numpy(), dtype=np.float32)
    new_dense = out[:n_dense] if n_dense > 0 else np.zeros((0, 3), dtype=np.float32)
    new_internal = out[n_dense:n_dense + n_internal] if n_internal > 0 else np.zeros((0, 3), dtype=np.float32)
    return new_dense, new_internal


def fill_particles_gaussian(
    ply_path,
    grid_n=64,
    max_samples=500_000,
    padding=0.1,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    use_batched_density=False,
    use_taichi=None,
    seed=None,
    opacity_thres=0.1,
):
    """
    PhysFlow-style filling from a 3DGS PLY (e.g. gaussian_output/{case}/{exp}/point_cloud/iteration_10000/point_cloud.ply).

    Returns:
        filled_pos: (N_orig + N_new, 3) tensor; first N_orig are Gaussian positions, then filled.
        n_new: number of filled particles added (dense + internal).
        new_internal_pos: (N_internal, 3) numpy; internal-filled particles only.
        new_dense_pos: (N_dense, 3) numpy; dense shell particles only. For MPM: init_pos = pkl + new_dense_pos + new_internal_pos.

    use_taichi: If True, use Taichi kernels (faster). If False, use NumPy. If None, use Taichi when available.
    """
    if PlyData is None:
        raise ImportError("plyfile is required: pip install plyfile")
    pos, opacity, cov = load_gaussian_ply(ply_path)
    pos = np.asarray(pos, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    mask = opacity > opacity_thres
    pos = pos[mask]
    opacity = opacity[mask]
    cov = cov[mask]
    if len(pos) == 0:
        raise ValueError(f"No Gaussians with opacity > {opacity_thres}")

    p_min = pos.min(axis=0)
    p_max = pos.max(axis=0)
    extent = (p_max - p_min).max()
    pad = extent * padding
    grid_origin = p_min - pad
    grid_lim = extent + 2 * pad
    grid_dx = grid_lim / grid_n
    pos_shifted = pos - grid_origin

    use_ti = (use_taichi if use_taichi is not None else TI_AVAILABLE)

    if use_ti and TI_AVAILABLE:
        new_dense, new_internal = _fill_particles_gaussian_taichi(
            pos_shifted, opacity, cov, grid_dx, grid_origin, grid_n,
            max_samples, density_thres, search_thres, max_particles_per_cell,
            search_exclude_dir, ray_cast_dir, seed,
        )
        n_dense = len(new_dense)
        n_internal = len(new_internal)
    else:
        rng = np.random.default_rng(seed)
        grid = assign_pos_to_grid(pos_shifted, np.zeros(3), grid_dx, grid_n)
        if use_batched_density:
            grid_density = densify_grids_batched(pos_shifted, opacity, cov, grid_dx, np.zeros(3), grid_n)
        else:
            grid_density = densify_grids_numpy(pos_shifted, opacity, cov, grid_dx, np.zeros(3), grid_n)
        new_dense = fill_dense_grids_numpy(grid, grid_density, grid_dx, grid_origin, density_thres, max_particles_per_cell, rng)
        n_dense = len(new_dense)
        new_internal = internal_filling_numpy(
            grid, grid_density, grid_dx, grid_origin, search_thres,
            max_particles_per_cell, search_exclude_dir, ray_cast_dir, rng
        )
        n_internal = len(new_internal)

    n_new = n_dense + n_internal
    if n_new > max_samples:
        new_dense = new_dense[:max(0, max_samples - n_internal)]
        new_internal = new_internal[:max(0, max_samples - len(new_dense))]
        n_new = len(new_dense) + len(new_internal)

    all_new = np.concatenate([new_dense, new_internal], axis=0) if n_dense and n_internal else (new_dense if n_dense else new_internal)
    filled = np.concatenate([pos, all_new], axis=0).astype(np.float32)
    new_internal_pos = new_internal.astype(np.float32) if n_internal else np.zeros((0, 3), dtype=np.float32)
    new_dense_pos = new_dense.astype(np.float32) if n_dense else np.zeros((0, 3), dtype=np.float32)
    return torch.from_numpy(filled), n_new, new_internal_pos, new_dense_pos


def get_gaussian_ply_path(gaussian_output_dir, case_name, exp_name="default", iteration=10000):
    """
    Resolve path: gaussian_output_dir/{case_name}/{exp_name}/point_cloud/iteration_{iteration}/point_cloud.ply
    Tries exp_name, then any first subdir under case_name if exp_name not found.
    """
    case_dir = os.path.join(gaussian_output_dir, case_name)
    if not os.path.isdir(case_dir):
        return None
    to_try = [exp_name]
    if exp_name == "default":
        try:
            subdirs = [d for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d))]
            to_try = [exp_name] + [s for s in subdirs if s != exp_name][:3]
        except OSError:
            pass
    for exp in to_try:
        base = os.path.join(case_dir, exp, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        if os.path.isfile(base):
            return base
        pc_dir = os.path.join(case_dir, exp, "point_cloud")
        if os.path.isdir(pc_dir):
            for name in sorted(os.listdir(pc_dir), reverse=True):
                if name.startswith("iteration_"):
                    path = os.path.join(pc_dir, name, "point_cloud.ply")
                    if os.path.isfile(path):
                        return path
    return None
