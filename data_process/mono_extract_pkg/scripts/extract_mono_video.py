"""
Extract DeformMaster-format data from a single monocular RGB video.

Default mode: VGGT-Omega temporal depth and pose, with metric scale anchored
from MoGe2 frame 0. This avoids the temporal point-cloud jitter caused by
using MoGe2 depth independently on every frame.

Output matches recorded_data/<case>/ format (1 cam):
  color/0/*.png + 0.mp4, depth/0/*.npy, calibrate.pkl, metadata.json

Usage:
    # Default: VGGT temporal depth + MoGe2 frame-0 scale
    python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
        --video /path/to/video.mp4 \
        --output_dir recorded_data/my_mono_case

    # DA3 only:
    python ... --da3_only
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PKG_ROOT = Path(__file__).resolve().parent.parent  # mono_extract_pkg/
THIRD_PARTY_ROOT = PKG_ROOT / "third_party"
DEFAULT_VGGT_CHECKPOINT = PKG_ROOT / "checkpoints" / "vggt_omega_1b_512.pt"


def run_da3(image_paths, model_name="depth-anything/DA3-Large", device="cuda:0",
            da3_root=None):
    """Run official DA3 → depth + intrinsics + extrinsics (w2c)."""
    da3_root = Path(da3_root or THIRD_PARTY_ROOT / "depth-anything-3")
    if not da3_root.is_dir():
        raise FileNotFoundError(
            f"DA3 repository not found at {da3_root}. Run "
            "`git submodule update --init --recursive`."
        )
    sys.path.insert(0, str(da3_root / "src"))
    from depth_anything_3.api import DepthAnything3

    print(f"[DA3] Loading {model_name}...")
    model = DepthAnything3.from_pretrained(model_name).to(device=torch.device(device))

    print(f"[DA3] Inferring {len(image_paths)} frames...")
    prediction = model.inference(image_paths)

    depth = prediction.depth              # (N, H, W) float32 meters
    intrinsics = prediction.intrinsics    # (N, 3, 3)
    extrinsics = prediction.extrinsics    # (N, 3, 4) w2c

    # w2c (3×4) → c2w (4×4)
    c2w = np.zeros((len(extrinsics), 4, 4), dtype=np.float64)
    for i in range(len(extrinsics)):
        w2c = np.eye(4)
        w2c[:3, :] = extrinsics[i]
        c2w[i] = np.linalg.inv(w2c)

    print(f"  depth: {depth.shape}, c2w: {c2w.shape}")
    print(f"  depth range: [{depth[depth > 0].min():.3f}, {depth.max():.3f}]m")
    cam_drift = np.linalg.norm(c2w[-1, :3, 3] - c2w[0, :3, 3])
    print(f"  camera drift (f0→f{len(c2w)-1}): {cam_drift*1000:.1f}mm")

    del model
    torch.cuda.empty_cache()
    return depth, intrinsics, c2w


def run_vggt(image_paths, vggt_root, checkpoint, device="cuda:0",
             image_resolution=512, mode="balanced"):
    """Run VGGT-Omega → depth + intrinsics + extrinsics (w2c).

    Drop-in replacement for run_da3 (same return signature). Pose and depth
    come from a single joint feed-forward pass, so they share one
    self-consistent scale — unlike DA3 pose + MoGe depth, which only agree
    after a fragile single-frame scale fit. depth is in VGGT's own scale;
    the downstream MoGe alignment anchors it to metric.
    """
    import sys
    if not vggt_root or not checkpoint:
        raise ValueError(
            "VGGT-Omega requires --vggt_root and --vggt_checkpoint, or the "
            "DEFORMMASTER_VGGT_ROOT and DEFORMMASTER_VGGT_CHECKPOINT "
            "environment variables. Pass --da3_only to use DA3 instead."
        )
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    print(f"[VGGT] Loading {checkpoint}...")
    model = VGGTOmega().to(device=torch.device(device)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    print(f"[VGGT] Inferring {len(image_paths)} frames (single pass)...")
    images = load_and_preprocess_images(
        image_paths, mode=mode, image_resolution=image_resolution
    ).to(torch.device(device))
    with torch.inference_mode():
        pred = model(images)

    H, W = pred["images"].shape[-2:]
    extr, intr = encoding_to_camera(pred["pose_enc"], (H, W))  # (B,T,3,4) w2c, (B,T,3,3)
    extr = extr.float().cpu().numpy().squeeze(0)
    intrinsics = intr.float().cpu().numpy().squeeze(0)
    depth = pred["depth"].float().cpu().numpy().squeeze(0)     # (T,H,W,1)
    if depth.ndim == 4:
        depth = depth[..., 0]                                  # (T,H,W)

    c2w = np.zeros((len(extr), 4, 4), dtype=np.float64)
    for i in range(len(extr)):
        w2c = np.eye(4)
        w2c[:3, :] = extr[i]
        c2w[i] = np.linalg.inv(w2c)

    print(f"  depth: {depth.shape}, c2w: {c2w.shape}")
    print(f"  depth range: [{depth[depth > 0].min():.3f}, {depth.max():.3f}] (vggt scale)")
    cam_drift = np.linalg.norm(c2w[-1, :3, 3] - c2w[0, :3, 3])
    print(f"  camera drift (f0->f{len(c2w)-1}): {cam_drift:.6f} (VGGT scale)")

    del model
    torch.cuda.empty_cache()
    return depth, intrinsics, c2w


def run_moge(frames_rgb, target_hw, device="cuda:0"):
    """Run MoGe2 → metric per-frame depth."""
    moge_root = Path(os.environ.get(
        "DEFORMMASTER_MOGE_ROOT", THIRD_PARTY_ROOT / "moge2"
    ))
    if not moge_root.is_dir():
        raise FileNotFoundError(
            f"MoGe2 repository not found at {moge_root}. Run "
            "`git submodule update --init --recursive`."
        )
    sys.path.insert(0, str(moge_root))
    from moge.model.v2 import MoGeModel

    print(f"[MoGe2] Loading model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    T = len(frames_rgb)
    H_out, W_out = target_hw
    depth_maps = np.zeros((T, H_out, W_out), dtype=np.float32)
    intrinsics_all = np.zeros((T, 3, 3), dtype=np.float32)

    for t, frame in enumerate(frames_rgb):
        img_t = torch.tensor(frame / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            output = model.infer(img_t, resolution_level=9, use_fp16=True,
                                 apply_mask=True, force_projection=True)
        dep = output['depth'].cpu().numpy()
        K = output['intrinsics'].cpu().numpy()  # normalized [0,1]

        # Scale intrinsics to pixel coords at target size
        K_px = np.array([
            [K[0, 0] * W_out, 0, K[0, 2] * W_out],
            [0, K[1, 1] * H_out, K[1, 2] * H_out],
            [0, 0, 1],
        ], dtype=np.float32)

        if (H_out, W_out) != dep.shape[:2]:
            dep = cv2.resize(dep, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        depth_maps[t] = dep
        intrinsics_all[t] = K_px

        if t % 10 == 0 or t == T - 1:
            d_valid = dep[dep > 0.01]
            if len(d_valid) > 0:
                print(f"  Frame {t}/{T-1}: depth=[{d_valid.min():.3f}, {d_valid.max():.3f}]m")

    K_median = np.median(intrinsics_all, axis=0)
    print(f"  MoGe2 intrinsics: fx={K_median[0,0]:.1f} fy={K_median[1,1]:.1f} "
          f"cx={K_median[0,2]:.1f} cy={K_median[1,2]:.1f}")

    del model
    torch.cuda.empty_cache()
    return depth_maps, K_median


def align_scale(moge_depth, source_depth, source_name, n_samples=10000):
    """Estimate scale: moge_depth ≈ scale * source_depth.

    Returns scale such that source coords × scale ≈ MoGe coords.
    Apply the same scale to the source c2w translations.
    """
    valid = (moge_depth > 0.01) & (source_depth > 0.01) & \
            np.isfinite(moge_depth) & np.isfinite(source_depth)
    moge_z = moge_depth[valid]
    source_z = source_depth[valid]

    if len(moge_z) < 100:
        print("  [WARN] Too few valid points for scale, using 1.0")
        return 1.0

    if len(moge_z) > n_samples:
        idx = np.random.RandomState(42).choice(len(moge_z), n_samples, replace=False)
        moge_z = moge_z[idx]
        source_z = source_z[idx]

    scale = np.median(moge_z / source_z)
    residual = np.abs(moge_z - scale * source_z)
    print(f"  Scale {source_name}→MoGe: {scale:.4f}")
    print(f"  Residual: mean={residual.mean():.4f}m, median={np.median(residual):.4f}m")
    return scale


def rescale_c2w(c2w, scale):
    """Scale c2w translations by a global metric scale factor."""
    c2w_scaled = c2w.copy()
    c2w_scaled[:, :3, 3] *= scale
    return c2w_scaled


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to extract (default: all frames)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--da3_model', type=str, default='depth-anything/DA3-Large')
    parser.add_argument('--da3_root', type=str,
                        default=os.environ.get(
                            'DEFORMMASTER_DA3_ROOT',
                            str(THIRD_PARTY_ROOT / 'depth-anything-3')),
                        help='Path to the depth-anything-3 repository.')
    parser.add_argument('--vggt_root', type=str,
                        default=os.environ.get(
                            'DEFORMMASTER_VGGT_ROOT',
                            str(THIRD_PARTY_ROOT / 'vggt-omega')),
                        help='Path to the vggt-omega repo (added to sys.path).')
    parser.add_argument('--vggt_checkpoint', type=str,
                        default=os.environ.get(
                            'DEFORMMASTER_VGGT_CHECKPOINT',
                            str(DEFAULT_VGGT_CHECKPOINT)),
                        help='VGGT-Omega checkpoint path (default: '
                             'mono_extract_pkg/checkpoints/vggt_omega_1b_512.pt).')
    parser.add_argument('--da3_only', action='store_true',
                        help='Use DA3 instead of the default VGGT-Omega + MoGe2 pipeline.')
    parser.add_argument('--fps', type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_moge = not args.da3_only

    # === Step 1: Read video ===
    print("=" * 60)
    print("Step 1: Reading video")
    print("=" * 60)
    cap = cv2.VideoCapture(args.video)
    fps = args.fps or int(round(cap.get(cv2.CAP_PROP_FPS)))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(T_total, args.max_frames) if args.max_frames else T_total
    print(f"  {T_total} frames, {W}x{H} @ {fps}fps → using first {T}")

    frames = []
    for _ in range(T):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    T = len(frames)
    H, W = frames[0].shape[:2]

    # === Step 2: Save RGB ===
    print("\n" + "=" * 60)
    print("Step 2: Saving RGB frames")
    print("=" * 60)
    color_dir = output_dir / 'color' / '0'
    color_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(frames):
        cv2.imwrite(str(color_dir / f'{t}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    mp4_path = output_dir / 'color' / '0.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  {T} frames → {color_dir} + {mp4_path}")

    # === Step 3: Temporal reconstruction (poses + depth + intrinsics) ===
    if not args.da3_only:
        pose_source = "VGGT-Omega"
        print("\n" + "=" * 60)
        print("Step 3: VGGT-Omega temporal reconstruction")
        print("=" * 60)
        image_paths = [str(color_dir / f'{t}.png') for t in range(T)]
        source_depth, source_intrinsics, c2w = run_vggt(
            image_paths, args.vggt_root, args.vggt_checkpoint, device=args.device
        )
    else:
        pose_source = "DA3"
        print("\n" + "=" * 60)
        print("Step 3: DA3 inference")
        print("=" * 60)
        image_paths = [str(color_dir / f'{t}.png') for t in range(T)]
        source_depth, source_intrinsics, c2w = run_da3(
            image_paths, model_name=args.da3_model, device=args.device,
            da3_root=args.da3_root
        )

    # === Step 3.5: Re-anchor poses so world frame = frame-0 camera ===
    # The reconstruction model returns poses in its own world frame, whose
    # origin/orientation is not guaranteed to coincide with the first camera. Downstream assumes
    # frame 0 sits at the world origin: the T_flip Z-down convention below and
    # calibrate.pkl's reference pose both rely on it. Left-multiply by inv(c2w[0])
    # so c2w[0] == I and every later frame encodes camera motion *relative to
    # frame 0*. Done before scale alignment so the rescale acts on
    # frame-0-relative translations.
    c2w = np.linalg.inv(c2w[0]) @ c2w
    drift = np.linalg.norm(c2w[-1, :3, 3] - c2w[0, :3, 3])
    print(f"  [frame0-relative] world frame = frame-0 camera; "
          f"drift f0->f{len(c2w)-1} = {drift*1000:.1f}mm (pre-scale)")

    # === Step 4: Depth + scale alignment ===
    if not args.da3_only:
        print("\n" + "=" * 60)
        print("Step 4: Metric scale alignment (VGGT-Omega + MoGe2)")
        print("=" * 60)
        # Only frame 0 of MoGe is needed (to anchor metric scale).
        moge_depth0, moge_K = run_moge(frames[:1], (H, W), device=args.device)

        # Resize VGGT depth to video resolution
        vggt_resized = source_depth
        if source_depth.shape[1:] != (H, W):
            vggt_resized = np.zeros((T, H, W), dtype=np.float32)
            for t in range(T):
                vggt_resized[t] = cv2.resize(source_depth[t], (W, H),
                                             interpolation=cv2.INTER_LINEAR)

        # One global scale from frame 0: depth_metric ≈ scale * depth_vggt.
        # VGGT depth is temporally consistent, so a single frame-0 scale keeps
        # the whole sequence metric AND jitter-free (unlike per-frame MoGe).
        print("\n  Aligning VGGT-Omega to MoGe2 metric scale...")
        scale = align_scale(moge_depth0[0], vggt_resized[0], "VGGT")
        depth_maps = vggt_resized * scale
        c2w = rescale_c2w(c2w, scale)
        K = moge_K.astype(np.float64)
        depth_source = "VGGT-Omega + MoGe2 metric scale"
    else:
        print("\n" + "=" * 60)
        print("Step 4: Using DA3 depth (da3_only mode)")
        print("=" * 60)
        depth_maps = source_depth
        K = source_intrinsics[0].astype(np.float64)
        depth_source = "DA3"

    # Resize depth to video resolution if needed
    if depth_maps.shape[1:] != (H, W):
        print(f"  Resizing depth {depth_maps.shape[1:]} → ({H}, {W})")
        resized = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            resized[t] = cv2.resize(depth_maps[t], (W, H), interpolation=cv2.INTER_LINEAR)
        depth_maps = resized
        scale_x = W / source_depth.shape[2]
        scale_y = H / source_depth.shape[1]
        K[0, 0] *= scale_x; K[0, 2] *= scale_x
        K[1, 1] *= scale_y; K[1, 2] *= scale_y

    # Save depth
    depth_dir = output_dir / 'depth' / '0'
    depth_dir.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        depth_mm = np.clip(depth_maps[t] * 1000.0, 0, 65535).astype(np.uint16)
        np.save(str(depth_dir / f'{t}.npy'), depth_mm)

    # === Step 5: Package ===
    print("\n" + "=" * 60)
    print("Step 5: Packaging")
    print("=" * 60)

    # Downstream pipeline (data_process_sample.py line 60) assumes Z-down world
    # (ground at z=0, objects above ground have z<0). OpenCV camera frame has
    # +Z pointing forward (into scene), so without a flip all depth points land
    # at z>0 and get clamped to the ground plane. Apply R_x(180°) so camera
    # forward → world -Z and camera +Y (down) → world +Y (still down, ground
    # plane). Valid for any frame-0-relative c2w sequence.
    T_flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    c2w = T_flip @ c2w

    c2w_0 = c2w[0].astype(np.float64)
    with open(str(output_dir / 'calibrate.pkl'), 'wb') as f:
        pickle.dump([c2w_0], f)

    np.save(str(output_dir / 'c2w_sequence.npy'), c2w)

    print(f"  Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    model_id = pose_source.lower().replace("-", "_")
    if use_moge:
        model_id += "_moge2"
    metadata = {
        "intrinsics": [K.tolist()],
        "serial_numbers": [f"mono_{model_id}"],
        "fps": fps,
        "WH": [W, H],
        "frame_num": T,
    }
    with open(str(output_dir / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Output: {output_dir}")
    print(f"  color/0/  : {T} PNG + MP4 ({W}x{H})")
    print(f"  depth/0/  : {T} NPY (uint16 mm, {depth_source})")
    scale_status = "MoGe metric scale" if use_moge else "original scale"
    print(f"  calibrate.pkl : 1 c2w ({pose_source}, {scale_status})")
    print(f"  c2w_sequence.npy : {T} per-frame c2w")


if __name__ == '__main__':
    main()
