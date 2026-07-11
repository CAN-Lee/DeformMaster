import warp as wp
import warp.torch
import torch


@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    # add
    mu_N: wp.array(dtype=float)
    lam_N: wp.array(dtype=float)
    viscosity: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    plastic_viscosity: wp.array(dtype=float)

    friction_angle: wp.array(dtype=float)
    alpha: float
    gravitational_accelaration: wp.vec3
    hardening: float
    xi: float  # wp.array(dtype=float)
    softening: wp.array(dtype=float)

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    particle_cov: wp.array(dtype=float)  # current covariance matrix
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    # add
    particle_F_N: wp.array(dtype=wp.mat33)
    particle_F_N_trial: wp.array(dtype=wp.mat33)
    particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)

    particle_selection: wp.array(
        dtype=int
    )  # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2

@wp.struct
class GridCollider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float
    mask: wp.array(dtype=int, ndim=3)


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float


@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float


@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)


@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


@wp.kernel
def mul_vec3_scalar(
    target_array: wp.array(dtype=wp.vec3), scale: float
):
    tid = wp.tid()
    target_array[tid] = target_array[tid] * scale


@wp.kernel
def mul_vec3_vec3_elementwise(
    target_array: wp.array(dtype=wp.vec3), scale_array: wp.array(dtype=float)
):
    tid = wp.tid()
    target_array[tid] = target_array[tid] * scale_array[tid]


@wp.kernel()
def set_value_mat33(x: wp.array(dtype=wp.mat33), y: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    x[tid] = y[tid]


@wp.kernel
def apply_pd_forces_kernel(
    particle_v: wp.array(dtype=wp.vec3),
    particle_x: wp.array(dtype=wp.vec3),
    controller_pos: wp.array(dtype=wp.vec3),
    controller_vel: wp.array(dtype=wp.vec3),
    controller_indices: wp.array(dtype=int, ndim=2), # [N_ctrl, K]
    controller_mask: wp.array(dtype=int, ndim=2),    # [N_ctrl, K]
    initial_offsets: wp.array(dtype=wp.vec3, ndim=2), # [N_ctrl, K]
    count_per_particle: wp.array(dtype=float),       # [N_sim]
    kp: float,
    kd: float,
    dt: float,
    clamp_dv: float
):
    # One thread per controller point and neighbor
    c_idx, k_idx = wp.tid()
    
    if controller_mask[c_idx, k_idx] == 0:
        return
        
    p_idx = controller_indices[c_idx, k_idx]
    
    target_pos = controller_pos[c_idx]
    target_vel = controller_vel[c_idx]
    
    curr_x = particle_x[p_idx]
    curr_v = particle_v[p_idx]
    
    curr_offset = target_pos - curr_x
    init_offset = initial_offsets[c_idx, k_idx]
    
    accel = kp * (curr_offset - init_offset) + kd * (target_vel - curr_v)
    dv = accel * dt
    
    # Soft normalization logic: total_dv = sum(dv) / sqrt(count)
    # We use atomic add to accumulate.
    # To implement the soft norm, we'll accumulate sum(dv) and then divide in a separate kernel, 
    # or just accumulate dv / sqrt(count) here.
    # Note: count_per_particle is fixed at reset.
    
    normalized_dv = dv / wp.pow(count_per_particle[p_idx], 0.5)
    
    # Atomic add to global particle velocity
    # We need to be careful about clamping. Clamping should happen on the total dV.
    # So we'll use a temporary array for total_dv.
    # Wait, for simplicity, let's just do atomic add to a temporary buffer.
    # For now, I'll just add directly to v, acknowledging that clamping might be slightly different.
    # Actually, let's add to a temp dv array.
    
@wp.kernel
def accumulate_pd_forces_kernel(
    particle_dv: wp.array(dtype=wp.vec3),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    controller_pos: wp.array(dtype=wp.vec3),
    controller_vel: wp.array(dtype=wp.vec3),
    controller_indices: wp.array(dtype=int, ndim=2),
    controller_mask: wp.array(dtype=int, ndim=2),
    initial_offsets: wp.array(dtype=wp.vec3, ndim=2),
    count_per_particle: wp.array(dtype=float),
    kp_arr: wp.array(dtype=float),
    kd_arr: wp.array(dtype=float),
    dt: float
):
    c_idx, k_idx = wp.tid()
    if controller_mask[c_idx, k_idx] == 0:
        return
    p_idx = controller_indices[c_idx, k_idx]
    
    kp_val = kp_arr[0]
    kd_val = kd_arr[0]
    accel = kp_val * (controller_pos[c_idx] - particle_x[p_idx] - initial_offsets[c_idx, k_idx]) + kd_val * (controller_vel[c_idx] - particle_v[p_idx])
    dv = (accel * dt) / wp.pow(count_per_particle[p_idx], 0.5)
    
    wp.atomic_add(particle_dv, p_idx, dv)

@wp.kernel
def apply_dv_with_clamping_kernel(
    particle_v: wp.array(dtype=wp.vec3),
    particle_dv: wp.array(dtype=wp.vec3),
    clamp_dv: float
):
    p = wp.tid()
    dv = particle_dv[p]
    
    # Manual vec3 clamping
    dv_mag = wp.length(dv)
    if dv_mag > clamp_dv:
        dv = dv * (clamp_dv / dv_mag)
        
    particle_v[p] = particle_v[p] + dv


def torch2warp_quat(t, copy=False, dtype=None, dvc="cuda:0", requires_grad=False):
    return wp.from_torch(t, dtype=wp.quat, requires_grad=requires_grad)

def torch2warp_float(t, copy=False, dtype=None, dvc="cuda:0", requires_grad=False):
    return wp.from_torch(t, dtype=wp.float32, requires_grad=requires_grad)

def torch2warp_vec3(t, copy=False, dtype=None, dvc="cuda:0", requires_grad=False):
    return wp.from_torch(t, dtype=wp.vec3, requires_grad=requires_grad)

def torch2warp_mat33(t, copy=False, dtype=None, dvc="cuda:0", requires_grad=False):
    # Reshape if necessary, ensuring it's [N, 3, 3] for mat33
    if t.shape[-1] == 9:
        t = t.view(-1, 3, 3)
    return wp.from_torch(t, dtype=wp.mat33, requires_grad=requires_grad)
