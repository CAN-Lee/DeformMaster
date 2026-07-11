"""
Multi-camera extrinsic calibration using a ChArUco board.

Takes one RGB image per camera (all seeing the same board), detects
ChArUco corners, solves PnP, and outputs calibrate.pkl (list of 3-4
camera-to-world 4x4 matrices).

Usage:
    # From pre-saved images
    python data_process/calibrate_charuco.py \
        --images cam0.png cam1.png cam2.png \
        --intrinsics intrinsics.json \
        --output data/different_types/my_case/calibrate.pkl

    # Or specify a directory with 0.png, 1.png, 2.png
    python data_process/calibrate_charuco.py \
        --image-dir data/different_types/my_case/calib_frames/ \
        --intrinsics intrinsics.json \
        --output data/different_types/my_case/calibrate.pkl

Board parameters (edit these to match YOUR printed board):
    CHARUCO_COLS, CHARUCO_ROWS, SQUARE_SIZE, MARKER_SIZE, ARUCO_DICT

intrinsics.json format:
    [
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   // cam 0
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   // cam 1
        ...
    ]
"""

import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np
from scipy.optimize import least_squares

# ============================================================
# Board parameters — EDIT THESE to match your printed board
# ============================================================
CHARUCO_COLS = 6          # number of squares horizontally
CHARUCO_ROWS = 9          # number of squares vertically
SQUARE_SIZE = 0.047       # square side length in meters (47 mm) — edit to measured value
MARKER_SIZE = 0.035       # ArUco marker side length in meters (35 mm) — edit to measured value
ARUCO_DICT = cv2.aruco.DICT_4X4_100  # must match the board PDF
# ============================================================


def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary,
    )
    return board, dictionary


def detect_and_solve(image_path, K, dist, board, dictionary):
    """Detect ChArUco corners and solve PnP for one camera.

    Returns:
        T_board2cam: 4x4 numpy array (board-to-camera transform)
        n_corners: number of detected corners
        reproj_error: mean reprojection error in pixels
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(img)

    if ch_corners is None or len(ch_corners) < 6:
        n = 0 if ch_corners is None else len(ch_corners)
        raise RuntimeError(
            f"{image_path}: only {n} ChArUco corners found (need ≥6)"
        )

    # Solve PnP
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)

    if not success:
        raise RuntimeError(f"{image_path}: solvePnP failed")

    # Reprojection error
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    error = float(np.sqrt(np.mean((projected.squeeze() - img_pts.squeeze()) ** 2)))

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T, len(ch_corners), error


def detect_charuco(image_path, board, dictionary):
    """Detect ChArUco corners (no PnP). Returns (obj_pts_3D, img_pts_2D, n_corners)
    or None if detection too weak. Uses OpenCV 4.7+ CharucoDetector API."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(img)
    if ch_corners is None or len(ch_corners) < 6:
        return None

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    return obj_pts.reshape(-1, 3), img_pts.reshape(-1, 2), int(len(ch_corners))


def rodrigues_to_mat(rvec, tvec):
    """6-DOF (rvec, tvec) → 4x4 SE(3)."""
    R, _ = cv2.Rodrigues(rvec.astype(np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.astype(np.float64).flatten()
    return T


def mat_to_rodrigues(T):
    """4x4 SE(3) → (rvec, tvec) each shape (3,)."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.flatten(), T[:3, 3].copy()


def project_batch(obj_pts_board, T_wb_k, T_cw_c, K):
    """Project board-frame points through pose T_wb_k (board_k→world) then camera T_cw_c (world→cam).
    T_wb_k, T_cw_c: 4x4 numpy arrays. Returns (N,2) pixel coords.
    """
    # p_world = T_wb_k @ p_board   (board_k point → world)
    # p_cam   = T_cw_c @ p_world   (world point → camera)
    N = obj_pts_board.shape[0]
    ones = np.ones((N, 1))
    p_b = np.hstack([obj_pts_board, ones])            # (N,4)
    p_w = (T_wb_k @ p_b.T).T                          # (N,4)
    p_c = (T_cw_c @ p_w.T).T[:, :3]                   # (N,3)
    # perspective divide + intrinsic
    uv = (K @ p_c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def bundle_adjust(observations, intrinsics_list, n_cams, n_poses, init_cams, init_poses,
                  verbose=True):
    """
    observations: dict (cam_idx, pose_idx) -> (obj_pts_board(N,3), img_pts(N,2))
    intrinsics_list: list of 3x3 K matrices, one per camera
    init_cams: list of 4x4 (board_0 → cam_c) transforms, length n_cams
    init_poses: list of 4x4 (board_k → board_0 = world) transforms, length n_poses
                init_poses[0] must be identity (gauge)

    Returns: (refined_cams, refined_poses, final_mean_reproj_px)
    """
    # Pack params: n_cams × 6 (rvec,tvec) + (n_poses-1) × 6  (pose 0 is identity, skipped)
    x0 = []
    for T in init_cams:
        rv, tv = mat_to_rodrigues(T)
        x0.extend(rv); x0.extend(tv)
    for T in init_poses[1:]:
        rv, tv = mat_to_rodrigues(T)
        x0.extend(rv); x0.extend(tv)
    x0 = np.array(x0, dtype=np.float64)

    def unpack(x):
        cams = []
        for c in range(n_cams):
            rv = x[c*6:c*6+3]
            tv = x[c*6+3:c*6+6]
            cams.append(rodrigues_to_mat(rv, tv))
        poses = [np.eye(4)]
        off = n_cams * 6
        for k in range(n_poses - 1):
            rv = x[off + k*6:off + k*6+3]
            tv = x[off + k*6+3:off + k*6+6]
            poses.append(rodrigues_to_mat(rv, tv))
        return cams, poses

    def residuals(x):
        cams, poses = unpack(x)
        res = []
        for (c, k), (obj_pts, img_pts) in observations.items():
            uv_pred = project_batch(obj_pts, poses[k], cams[c], intrinsics_list[c])
            res.append((uv_pred - img_pts).flatten())
        return np.concatenate(res)

    # Initial residual
    r0 = residuals(x0)
    initial_rms = float(np.sqrt(np.mean(r0**2)))
    if verbose:
        print(f"  initial reproj RMS: {initial_rms:.3f} px "
              f"({len(r0)//2} observations)")

    # Optimize (trust-region with numeric Jacobian)
    result = least_squares(residuals, x0, method='trf', verbose=2 if verbose else 0,
                           max_nfev=200)
    final_rms = float(np.sqrt(np.mean(result.fun**2)))
    if verbose:
        print(f"  final reproj RMS:   {final_rms:.3f} px")

    cams, poses = unpack(result.x)
    return cams, poses, final_rms


def run_multi_pose_calibration(pose_dir, intrinsics_list, args):
    """Joint bundle adjustment over multiple board poses."""
    # Discover pose subdirs: pose_0/, pose_1/, ...
    pose_dirs = sorted(glob.glob(os.path.join(pose_dir, "pose_*")))
    pose_dirs = [p for p in pose_dirs if os.path.isdir(p)]
    if len(pose_dirs) < 2:
        raise RuntimeError(f"Need ≥2 pose subdirs in {pose_dir}, found {len(pose_dirs)}")

    # Probe n_cams from pose_0
    cam_images_0 = sorted(glob.glob(os.path.join(pose_dirs[0], "*.png")))
    n_cams = len(cam_images_0)
    if n_cams != len(intrinsics_list):
        raise ValueError(f"{n_cams} images in {pose_dirs[0]} but "
                         f"{len(intrinsics_list)} intrinsics entries")

    n_poses = len(pose_dirs)
    print(f"Multi-pose calibration: {n_cams} cameras × {n_poses} poses")
    print(f"Board: {CHARUCO_COLS}x{CHARUCO_ROWS}, square={SQUARE_SIZE*1000:.0f}mm, "
          f"marker={MARKER_SIZE*1000:.0f}mm, dict=DICT_4X4\n")

    board, dictionary = create_board()
    dist = np.zeros(5)

    # 1. Detect corners in every (camera, pose) image
    observations = {}  # (cam_idx, pose_idx) -> (obj_pts, img_pts)
    detection_table = np.zeros((n_cams, n_poses), dtype=np.int32)
    for k, pdir in enumerate(pose_dirs):
        for c in range(n_cams):
            img_path = os.path.join(pdir, f"{c}.png")
            if not os.path.exists(img_path):
                continue
            result = detect_charuco(img_path, board, dictionary)
            if result is None:
                continue
            obj_pts, img_pts, n_corners = result
            observations[(c, k)] = (obj_pts, img_pts)
            detection_table[c, k] = n_corners

    # Print detection table
    print("Detection matrix (rows=cameras, cols=poses, values=corner counts):")
    print("        " + "  ".join(f"p{k}" for k in range(n_poses)))
    for c in range(n_cams):
        row = "  ".join(f"{detection_table[c,k]:3d}" for k in range(n_poses))
        print(f"  cam{c}: {row}")
    print()

    # 2. Initialize camera extrinsics using pose_0 PnP
    print("Initializing camera extrinsics from pose_0...")
    init_cams = []
    for c in range(n_cams):
        if (c, 0) not in observations:
            raise RuntimeError(f"cam {c} did not see pose_0 — need it for gauge init")
        obj_pts, img_pts = observations[(c, 0)]
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsics_list[c], dist)
        if not ok:
            raise RuntimeError(f"cam {c} pose_0 PnP failed")
        T = rodrigues_to_mat(rvec, tvec)
        init_cams.append(T)
        print(f"  cam {c}: init distance to board = {np.linalg.norm(T[:3,3]):.3f} m")

    # 3. Initialize board-pose transforms using any camera that saw both pose_0 and pose_k
    print("\nInitializing board pose transforms...")
    init_poses = [np.eye(4)]  # pose 0 = world gauge
    for k in range(1, n_poses):
        T_wb_k = None
        for c in range(n_cams):
            if (c, k) not in observations:
                continue
            obj_pts, img_pts = observations[(c, k)]
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsics_list[c], dist)
            if not ok:
                continue
            T_bc_k = rodrigues_to_mat(rvec, tvec)        # board_k → cam_c
            # T_wb_k = T_cw_c^{-1} @ T_bc_k, where T_cw_c = init_cams[c] is board_0→cam_c
            T_wb_k = np.linalg.inv(init_cams[c]) @ T_bc_k
            break
        if T_wb_k is None:
            raise RuntimeError(f"pose {k} not seen by any camera")
        init_poses.append(T_wb_k)
        print(f"  pose {k}: init translation = "
              f"[{T_wb_k[0,3]:+.3f}, {T_wb_k[1,3]:+.3f}, {T_wb_k[2,3]:+.3f}]")

    # 4. Bundle adjust
    print("\nBundle adjustment...")
    refined_cams, refined_poses, final_rms = bundle_adjust(
        observations, intrinsics_list, n_cams, n_poses,
        init_cams, init_poses, verbose=True,
    )

    # 5. Output c2w (board_0 / world → cam_c, then invert)
    c2ws = [np.linalg.inv(T) for T in refined_cams]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(c2ws, f)

    print(f"\nSaved {n_cams} c2w matrices to {args.output}")
    print(f"Final reproj RMS: {final_rms:.3f} px\n")
    for i, c2w in enumerate(c2ws):
        pos = c2w[:3, 3]
        print(f"  cam {i} position: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}], "
              f"distance to board: {np.linalg.norm(pos):.3f} m")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--images", nargs="+",
        help="Paths to calibration images, one per camera (in order). Single-pose mode."
    )
    group.add_argument(
        "--image-dir",
        help="Directory containing 0.png, 1.png, 2.png, ... Single-pose mode."
    )
    group.add_argument(
        "--pose-dir",
        help="Directory containing pose_0/, pose_1/, ... subdirs, each with "
             "per-camera PNGs. Multi-pose bundle-adjustment mode."
    )
    parser.add_argument(
        "--intrinsics", required=True,
        help="Path to intrinsics JSON (list of 3x3 K matrices)"
    )
    parser.add_argument(
        "--output", default="calibrate.pkl",
        help="Output path for calibrate.pkl"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show detected corners overlaid on images"
    )
    args = parser.parse_args()

    # Load intrinsics
    with open(args.intrinsics) as f:
        intrinsics_list = json.load(f)
    intrinsics_list = [np.array(K, dtype=np.float64) for K in intrinsics_list]

    # Multi-pose bundle adjustment path
    if args.pose_dir:
        run_multi_pose_calibration(args.pose_dir, intrinsics_list, args)
        return

    # Gather image paths (single-pose path)
    if args.images:
        image_paths = args.images
    else:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
        if not image_paths:
            image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))

    n_cams = len(image_paths)
    if n_cams != len(intrinsics_list):
        raise ValueError(
            f"{n_cams} images but {len(intrinsics_list)} intrinsics entries"
        )

    print(f"Calibrating {n_cams} cameras")
    print(f"Board: {CHARUCO_COLS}x{CHARUCO_ROWS}, square={SQUARE_SIZE*1000:.0f}mm, "
          f"marker={MARKER_SIZE*1000:.0f}mm, dict=DICT_4X4")
    print()

    board, dictionary = create_board()
    dist = np.zeros(5)  # assume no distortion (or pass via arg)

    T_board2cams = []
    for i, (img_path, K) in enumerate(zip(image_paths, intrinsics_list)):
        try:
            T, n_corners, error = detect_and_solve(img_path, K, dist, board, dictionary)
            T_board2cams.append(T)
            print(f"  cam {i}: {n_corners} corners, reproj error = {error:.3f} px  ✓")

            if error > 1.0:
                print(f"    ⚠️  reproj error > 1px, check image quality or board flatness")

        except RuntimeError as e:
            print(f"  cam {i}: FAILED — {e}")
            return

    # Convert to c2w: board frame = world frame
    # c2w = inv(board2cam)
    c2ws = [np.linalg.inv(T) for T in T_board2cams]

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(c2ws, f)

    print()
    print(f"Saved {n_cams} c2w matrices to {args.output}")
    print()

    # Print translation summary (sanity check: cameras should be ~0.5-1.5m from board)
    for i, c2w in enumerate(c2ws):
        pos = c2w[:3, 3]
        dist_to_board = np.linalg.norm(pos)
        print(f"  cam {i} position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
              f"distance to board: {dist_to_board:.3f} m")

    if args.visualize:
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            corners, ids, _ = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters()
            ).detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.imshow(f"cam {i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
