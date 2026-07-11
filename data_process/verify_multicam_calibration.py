"""Quick sanity check for multi-cam calibration + recording.

Loads one frame from all cameras, back-projects depth to 3D using each
cam's intrinsics + c2w extrinsic, and displays the 5 point clouds
together in Open3D. If calibration is correct, the 5 clouds overlap into
one coherent scene. If any cam is misaligned, its cloud floats off.

Two color modes:
  * default (--rgb): color each point with its actual RGB pixel — you see
    the real scene; correctness confirmed when everything looks like one
    scene rather than 5 duplicates.
  * --per-cam-color: one solid color per cam (red/green/blue/yellow/magenta)
    — fastest way to spot alignment errors.

Usage:
    python data_process/verify_multicam_calibration.py recorded_data/<case>
    python data_process/verify_multicam_calibration.py recorded_data/<case> \\
        --frame 0 --per-cam-color --max-depth 2.0
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np


def load_case(case_dir):
    with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    with open(os.path.join(case_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    Ks = [np.array(k, dtype=np.float64) for k in meta["intrinsics"]]
    assert len(c2ws) == len(Ks), \
        f"cam count mismatch: calibrate.pkl has {len(c2ws)} but metadata " \
        f"has {len(Ks)}"
    return c2ws, Ks, meta


def backproject(depth_mm, rgb, K, c2w, max_depth_m=2.5):
    """depth (H, W) uint16 mm → world points (N, 3), colors (N, 3) in [0,1]."""
    H, W = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) / 1000.0  # mm → m
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    mask = (depth_m > 0.1) & (depth_m < max_depth_m)
    u = u[mask]
    v = v[mask]
    d = depth_m[mask]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    pts_cam = np.stack([x, y, z], axis=1)   # (N, 3)
    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_world = (c2w @ pts_cam_h.T).T[:, :3]
    if rgb is not None:
        rgb_bgr = rgb
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb_rgb[v, u].astype(np.float32) / 255.0
    else:
        colors = None
    return pts_world, colors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir",
                        help="Path to recorded_data/<case_name>")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to visualize (default 0)")
    parser.add_argument("--max-depth", type=float, default=2.5,
                        help="Clip depth beyond this (m) to remove noise")
    parser.add_argument("--per-cam-color", action="store_true",
                        help="Color each cam's points with a solid color "
                             "(easier to spot misalignment)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Use every Nth pixel to speed up viz (default 2)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip Open3D window, just print stats")
    args = parser.parse_args()

    c2ws, Ks, meta = load_case(args.case_dir)
    n_cams = len(c2ws)
    print(f"Loaded {n_cams} cams from {args.case_dir}")
    print(f"  frame {args.frame}, depth clip {args.max_depth}m")

    # Solid per-cam colors (RGB in [0,1])
    CAM_COLORS = [
        [1.0, 0.2, 0.2],   # red
        [0.2, 1.0, 0.2],   # green
        [0.2, 0.4, 1.0],   # blue
        [1.0, 1.0, 0.2],   # yellow
        [1.0, 0.2, 1.0],   # magenta
    ]

    all_points = []
    all_colors = []
    centroids = []
    for i in range(n_cams):
        color_path = os.path.join(
            args.case_dir, "color", str(i), f"{args.frame}.png")
        depth_path = os.path.join(
            args.case_dir, "depth", str(i), f"{args.frame}.npy")
        if not os.path.exists(depth_path):
            print(f"  [WARN] cam {i} missing {depth_path}")
            continue
        depth = np.load(depth_path)
        rgb = cv2.imread(color_path) if os.path.exists(color_path) else None
        # Downsample for viz speed
        if args.downsample > 1:
            depth = depth[::args.downsample, ::args.downsample]
            if rgb is not None:
                rgb = rgb[::args.downsample, ::args.downsample]
            K = Ks[i].copy()
            K[0, 0] /= args.downsample   # fx
            K[1, 1] /= args.downsample   # fy
            K[0, 2] /= args.downsample   # cx
            K[1, 2] /= args.downsample   # cy
        else:
            K = Ks[i]
        pts, cols = backproject(depth, rgb, K, c2ws[i],
                                max_depth_m=args.max_depth)
        if pts.shape[0] == 0:
            print(f"  [WARN] cam {i} yielded 0 valid points")
            continue
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        print(f"  cam {i}: {pts.shape[0]:>6d} pts  "
              f"centroid=({centroid[0]:+.3f}, {centroid[1]:+.3f}, "
              f"{centroid[2]:+.3f})")
        all_points.append(pts)
        if args.per_cam_color or cols is None:
            cols = np.tile(CAM_COLORS[i % len(CAM_COLORS)],
                           (pts.shape[0], 1))
        all_colors.append(cols)

    if not all_points:
        print("[ERROR] no valid point clouds")
        sys.exit(1)

    # Print pairwise centroid distance — rough sanity (should be cluster <0.5m
    # for a table-top scene; if >1m, likely misalignment)
    print("\nPairwise centroid distances (meters):")
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            flag = " ← LARGE" if d > 0.8 else ""
            print(f"  cam{i}-cam{j}: {d:.3f}{flag}")

    if args.no_viz:
        return

    try:
        import open3d as o3d
    except ImportError:
        print("\n[ERROR] open3d not installed; skip visualization.")
        print("  pip install open3d    (or add --no-viz)")
        sys.exit(1)

    pcds = []
    for pts, cols in zip(all_points, all_colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcds.append(pcd)

    # Add world-frame coordinate axes
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])

    print("\nOpening Open3D viewer — close window when done.")
    print("  * Drag to rotate  * Scroll to zoom  * Shift+drag to pan")
    print("  * Good alignment = 5 clouds form one coherent scene")
    print("  * Bad  alignment = clouds float apart (use --per-cam-color "
          "to see)")
    o3d.visualization.draw_geometries([*pcds, frame],
                                      window_name=f"Multi-cam verify: "
                                      f"{os.path.basename(args.case_dir)}")


if __name__ == "__main__":
    main()
