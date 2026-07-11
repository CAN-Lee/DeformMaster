"""
Capture one synchronized RGB frame from each Orbbec Gemini camera +
dump intrinsics. First step of multi-camera calibration.

Star-hub hardware sync: PRIMARY device fires trigger, SECONDARY devices
capture on trigger with staggered delays to prevent IR interference.

Usage:
    # First run: discover devices and print serials
    python data_process/capture_calib_frames.py --list

    # Then set MASTER_SERIAL below (or pass --master-serial), and run:
    python data_process/capture_calib_frames.py --output-dir calib_frames/

Outputs:
    calib_frames/0.png ... N.png       (one RGB image per camera, ordered by serial)
    calib_frames/intrinsics.json       (list of K matrices, same order)
    calib_frames/serials.json          (camera index → serial mapping)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, Context,
    OBSensorType, OBFormat,
    OBMultiDeviceSyncMode,
)


# ============================================================
# Sync setup — EDIT MASTER_SERIAL after running --list once
# ============================================================
MASTER_SERIAL = "AY3A13100CM"  # the device cabled to hub IN
SECONDARY_DELAY_STEP_US = 4000  # 4ms between SECONDARYs (star hub IR de-conflict)
# ============================================================


def list_devices(ctx):
    """Print serials of all connected devices."""
    devices = ctx.query_devices()
    n = devices.get_count()
    print(f"Found {n} Orbbec devices:")
    for i in range(n):
        device = devices.get_device_by_index(i)
        sn = device.get_device_info().get_serial_number()
        name = device.get_device_info().get_name()
        print(f"  [{i}] serial={sn}  name={name}")
    print()
    print("Set MASTER_SERIAL in the script (or pass --master-serial) to "
          "the device cabled to your sync hub's IN port.")


def configure_sync(devices, master_serial):
    """Set PRIMARY/SECONDARY sync mode for each device. Returns ordered list
    of (device, is_master, secondary_index)."""
    n = devices.get_count()
    serials_in_order = []
    for i in range(n):
        device = devices.get_device_by_index(i)
        sn = device.get_device_info().get_serial_number()
        serials_in_order.append((device, sn))

    # Sort: master first, then slaves (consistent ordering)
    master_entry = None
    slaves = []
    for device, sn in serials_in_order:
        if sn == master_serial:
            master_entry = (device, sn)
        else:
            slaves.append((device, sn))

    if master_entry is None:
        raise RuntimeError(
            f"Master serial {master_serial} not found among connected devices: "
            f"{[s for _, s in serials_in_order]}"
        )

    # Configure master
    master_device, master_sn = master_entry
    sync_cfg = master_device.get_multi_device_sync_config()
    sync_cfg.mode = OBMultiDeviceSyncMode.PRIMARY
    sync_cfg.depth_delay_us = 0
    sync_cfg.color_delay_us = 0
    sync_cfg.trigger_to_image_delay_us = 0
    sync_cfg.trigger_out_enable = True
    sync_cfg.trigger_out_delay_us = -1
    sync_cfg.frames_per_trigger = 1
    master_device.set_multi_device_sync_config(sync_cfg)
    print(f"  [PRIMARY] {master_sn}")

    # Configure slaves with staggered delays
    for slave_idx, (device, sn) in enumerate(slaves):
        sync_cfg = device.get_multi_device_sync_config()
        sync_cfg.mode = OBMultiDeviceSyncMode.SECONDARY
        sync_cfg.depth_delay_us = 0
        sync_cfg.color_delay_us = 0
        # Stagger SECONDARYs to avoid IR projector interference
        sync_cfg.trigger_to_image_delay_us = (slave_idx + 1) * SECONDARY_DELAY_STEP_US
        sync_cfg.trigger_out_enable = False
        sync_cfg.frames_per_trigger = 1
        device.set_multi_device_sync_config(sync_cfg)
        print(f"  [SECONDARY {slave_idx}] {sn}  delay={sync_cfg.trigger_to_image_delay_us}us")

    # Return ordered list: [master, slave0, slave1, ...]
    return [master_entry] + slaves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="Just list connected devices and exit")
    parser.add_argument("--output-dir", default="calib_frames",
                        help="Where to save images + intrinsics.json")
    parser.add_argument("--master-serial", default=None,
                        help="Override MASTER_SERIAL constant")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip live preview window (single-shot fallback; implies --single-pose)")
    parser.add_argument("--single-pose", action="store_true",
                        help="Legacy mode: capture one pose only. Default is multi-pose BA.")
    args = parser.parse_args()
    # Multi-pose is the default; --single-pose or --no-preview opts out.
    args.multi_pose = not (args.single_pose or args.no_preview)

    ctx = Context()

    if args.list:
        list_devices(ctx)
        return

    master_serial = args.master_serial or MASTER_SERIAL
    if not master_serial:
        print("ERROR: MASTER_SERIAL not set.")
        print("Run with --list first to discover serials, then either edit")
        print("the script or pass --master-serial <serial>.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    devices = ctx.query_devices()
    n = devices.get_count()
    if n < 2:
        raise RuntimeError("Need ≥2 cameras for sync calibration")

    # 1. Configure sync modes (master + slaves with staggered delays)
    print(f"\nConfiguring sync (star hub, master={master_serial})...")
    ordered_devices = configure_sync(devices, master_serial)

    # 2. Build pipelines + read intrinsics
    pipelines = []
    intrinsics = []
    serials = []
    is_master_flags = []

    for idx, (device, sn) in enumerate(ordered_devices):
        is_master = (idx == 0)
        pipe = Pipeline(device)
        cfg = Config()
        profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(
            args.width, args.height, OBFormat.RGB, 30
        )
        cfg.enable_stream(color_profile)

        intr = color_profile.get_intrinsic()
        K = [
            [intr.fx, 0,       intr.cx],
            [0,       intr.fy, intr.cy],
            [0,       0,       1.0],
        ]
        intrinsics.append(K)
        serials.append(sn)
        is_master_flags.append(is_master)
        pipelines.append((pipe, cfg, is_master, sn))

    # 3. Start SECONDARY pipelines FIRST (they enter "wait-for-trigger" state)
    print("\nStarting SECONDARY pipelines first...")
    for pipe, cfg, is_master, sn in pipelines:
        if not is_master:
            pipe.start(cfg)
            print(f"  started slave {sn}")

    time.sleep(0.5)  # let slaves enter wait state

    # 4. Start PRIMARY last (it begins firing triggers)
    print("Starting PRIMARY pipeline (begins triggering)...")
    for pipe, cfg, is_master, sn in pipelines:
        if is_master:
            pipe.start(cfg)
            print(f"  started master {sn}")

    # 5. Warmup
    print(f"\nWarming up ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        for pipe, _, _, _ in pipelines:
            try:
                pipe.wait_for_frames(200)
            except Exception:
                pass

    # 6. Preview loop + capture on SPACE
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    def grab_bgr(pipe):
        frames = pipe.wait_for_frames(500)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        rgb = np.frombuffer(color_frame.get_data(), np.uint8).reshape(
            args.height, args.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    latest_bgr = [None] * len(pipelines)
    saved_poses = 0  # how many poses have been captured so far (multi-pose mode)

    def save_current_frames(pose_idx=None):
        """Save the latest frame from each camera. If pose_idx is given, save under pose_<idx>/."""
        if pose_idx is None:
            subdir = args.output_dir
        else:
            subdir = os.path.join(args.output_dir, f"pose_{pose_idx}")
            os.makedirs(subdir, exist_ok=True)
        for cam_idx, (pipe, _, is_master, sn) in enumerate(pipelines):
            bgr = latest_bgr[cam_idx] if not args.no_preview else grab_bgr(pipe)
            if bgr is None:
                print(f"  cam {cam_idx} ({sn}): TIMEOUT — no frame received")
                continue
            out_path = os.path.join(subdir, f"{cam_idx}.png")
            cv2.imwrite(out_path, bgr)
            role = "MASTER" if is_master else "slave "
            tag = f"pose_{pose_idx}/" if pose_idx is not None else ""
            print(f"  cam {cam_idx} [{role}] {sn}: saved {tag}{cam_idx}.png")

    if not args.no_preview:
        print("\nPreview mode — check all cameras see the ChArUco board.")
        if args.multi_pose:
            print("  [SPACE] save current pose   [ENTER] finish   [Q] abort")
        else:
            print("  [SPACE] capture      [Q] quit")

        n_cams = len(pipelines)
        n_cols = min(n_cams, 3)
        n_rows = (n_cams + n_cols - 1) // n_cols
        tile_w, tile_h = 640, 360

        while True:
            for cam_idx, (pipe, _, _, _) in enumerate(pipelines):
                bgr = grab_bgr(pipe)
                if bgr is not None:
                    latest_bgr[cam_idx] = bgr

            tiles = []
            for cam_idx, (_, _, is_master, sn) in enumerate(pipelines):
                bgr = latest_bgr[cam_idx]
                if bgr is None:
                    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    cv2.putText(tile, f"cam {cam_idx}: no frame",
                                (10, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                else:
                    tile = cv2.resize(bgr, (tile_w, tile_h))
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detector.detectMarkers(gray)
                    n_markers = 0 if ids is None else len(ids)
                    if n_markers > 0:
                        sx = tile_w / args.width
                        sy = tile_h / args.height
                        scaled = [c * np.array([[sx, sy]], dtype=np.float32)
                                  for c in corners]
                        cv2.aruco.drawDetectedMarkers(tile, scaled, ids)
                    role = "PRIMARY" if is_master else "secondary"
                    color = (0, 255, 0) if n_markers >= 10 else \
                            (0, 200, 255) if n_markers >= 4 else (0, 0, 255)
                    cv2.putText(tile, f"[{cam_idx}] {role} {sn[-4:]}  markers={n_markers}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2)
                tiles.append(tile)

            while len(tiles) < n_rows * n_cols:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

            rows = [np.hstack(tiles[r * n_cols:(r + 1) * n_cols])
                    for r in range(n_rows)]
            grid = np.vstack(rows)

            # Status bar at top for multi-pose mode
            if args.multi_pose:
                bar_h = 40
                bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar,
                            f"Saved poses: {saved_poses}   "
                            f"[SPACE] save   [ENTER] finish   [Q] abort",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                grid = np.vstack([bar, grid])

            cv2.imshow("calibration preview", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if args.multi_pose:
                    print(f"\n[SPACE] saving pose {saved_poses}...")
                    save_current_frames(pose_idx=saved_poses)
                    saved_poses += 1
                else:
                    print("\n[SPACE] captured — saving frames...")
                    save_current_frames()
                    break
            elif args.multi_pose and (key == 13 or key == 10):  # ENTER
                if saved_poses < 2:
                    print(f"\n[ENTER] need at least 2 poses (have {saved_poses}). "
                          "Press SPACE to save more.")
                    continue
                print(f"\n[ENTER] finished: {saved_poses} poses saved")
                break
            elif key == ord('q'):
                print("\n[Q] aborted by user")
                cv2.destroyAllWindows()
                for pipe, _, _, _ in pipelines:
                    pipe.stop()
                return

        cv2.destroyAllWindows()
    else:
        # no-preview path: single-shot only (multi-pose requires interaction)
        save_current_frames()

    # 7. Save intrinsics + serial mapping
    intr_path = os.path.join(args.output_dir, "intrinsics.json")
    with open(intr_path, "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"\nSaved intrinsics to {intr_path}")

    serial_path = os.path.join(args.output_dir, "serials.json")
    with open(serial_path, "w") as f:
        json.dump([{"index": i, "serial": s, "is_master": m}
                   for i, (s, m) in enumerate(zip(serials, is_master_flags))],
                  f, indent=2)
    print(f"Saved serial mapping to {serial_path}")

    # 8. Stop pipelines
    for pipe, _, _, _ in pipelines:
        pipe.stop()

    print("\nNext step:")
    if args.multi_pose:
        print(f"  python data_process/calibrate_charuco.py \\")
        print(f"      --pose-dir {args.output_dir}/ \\")
        print(f"      --intrinsics {intr_path} \\")
        print(f"      --output {args.output_dir}/calibrate.pkl")
    else:
        print(f"  python data_process/calibrate_charuco.py \\")
        print(f"      --image-dir {args.output_dir}/ \\")
        print(f"      --intrinsics {intr_path} \\")
        print(f"      --output {args.output_dir}/calibrate.pkl")


if __name__ == "__main__":
    main()
