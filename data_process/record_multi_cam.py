"""
Record synchronized RGBD frames from multiple Orbbec Gemini cameras.

Star-hub hardware sync: PRIMARY device fires trigger, SECONDARY devices
capture on trigger with staggered IR delays. Saves data in the format
expected by data_process/* pipeline:

    <output_dir>/
    ├── color/{0,1,2,3}/0000.png ...
    ├── depth/{0,1,2,3}/0000.npy ...   (uint16, mm)
    ├── calibrate.pkl                   (copied from calibration step)
    └── metadata.json                   (intrinsics + WH + frame_num)

Usage:
    # First-time setup: discover cameras (re-uses MASTER_SERIAL from
    # capture_calib_frames.py)
    python data_process/record_multi_cam.py --list

    # Record one case (300 frames @ 30fps = 10 seconds)
    python data_process/record_multi_cam.py \\
        --output recorded_data/fold_cloth_1 \\
        --calibrate-pkl calibrate.pkl \\
        --frames 300
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, Context,
    OBSensorType, OBFormat, OBStreamType,
    OBMultiDeviceSyncMode,
    AlignFilter,
)


# ============================================================
# Sync setup — keep in sync with capture_calib_frames.py
# ============================================================
MASTER_SERIAL = "AY3A13100CM"  # the device cabled to hub IN
SECONDARY_DELAY_STEP_US = 4000
# ============================================================


def list_devices(ctx):
    devices = ctx.query_devices()
    n = devices.get_count()
    print(f"Found {n} Orbbec devices:")
    for i in range(n):
        device = devices.get_device_by_index(i)
        sn = device.get_device_info().get_serial_number()
        name = device.get_device_info().get_name()
        print(f"  [{i}] serial={sn}  name={name}")


def configure_sync(devices, master_serial):
    """Set PRIMARY/SECONDARY modes. Returns ordered [(device, serial), ...]."""
    n = devices.get_count()
    entries = [(devices.get_device_by_index(i),
                devices.get_device_by_index(i).get_device_info().get_serial_number())
               for i in range(n)]

    master_entry = next((e for e in entries if e[1] == master_serial), None)
    if master_entry is None:
        raise RuntimeError(
            f"Master serial {master_serial} not found among "
            f"{[s for _, s in entries]}"
        )
    slaves = [e for e in entries if e[1] != master_serial]

    # Master
    master_dev = master_entry[0]
    cfg = master_dev.get_multi_device_sync_config()
    cfg.mode = OBMultiDeviceSyncMode.PRIMARY
    cfg.depth_delay_us = 0
    cfg.color_delay_us = 0
    cfg.trigger_to_image_delay_us = 0
    cfg.trigger_out_enable = True
    cfg.trigger_out_delay_us = -1
    cfg.frames_per_trigger = 1
    master_dev.set_multi_device_sync_config(cfg)
    print(f"  [PRIMARY] {master_entry[1]}")

    # Slaves with staggered IR delays
    for slave_idx, (device, sn) in enumerate(slaves):
        cfg = device.get_multi_device_sync_config()
        cfg.mode = OBMultiDeviceSyncMode.SECONDARY
        cfg.depth_delay_us = 0
        cfg.color_delay_us = 0
        cfg.trigger_to_image_delay_us = (slave_idx + 1) * SECONDARY_DELAY_STEP_US
        cfg.trigger_out_enable = False
        cfg.frames_per_trigger = 1
        device.set_multi_device_sync_config(cfg)
        print(f"  [SECONDARY {slave_idx}] {sn}  delay={cfg.trigger_to_image_delay_us}us")

    return [master_entry] + slaves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--master-serial", default=None)
    parser.add_argument("--output", default=None,
                        help="Case output dir (e.g., recorded_data/fold_cloth_1)")
    parser.add_argument("--calibrate-pkl", default=None,
                        help="Path to calibrate.pkl from calibration step "
                             "(copied to output dir)")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to record (default 300 = 10s@30fps)")
    parser.add_argument("--width", type=int, default=640,
                        help="Color resolution width (default 640). "
                             "Gemini 2 supports 640x360 / 640x480 / 1280x720 "
                             "/ 1920x1080 (no native 848x480).")
    parser.add_argument("--height", type=int, default=360,
                        help="Color resolution height (default 360). "
                             "640x360 is 16:9 and minimizes USB bandwidth.")
    parser.add_argument("--depth-width", type=int, default=640,
                        help="Depth native width. Gemini 2 supports "
                             "320x200 / 640x400 / 1280x800. "
                             "AlignFilter warps depth to color resolution, "
                             "so native 640x400 (16:9-ish) is plenty for "
                             "640x360 color.")
    parser.add_argument("--depth-height", type=int, default=400,
                        help="Depth native height (default 400)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--start-delay", type=float, default=0.0,
                        help="Seconds to wait before recording starts (after "
                             "warmup). Use to walk to the scene.")
    parser.add_argument("--preview", action="store_true",
                        help="Show 5-tile live preview window. Full-rate "
                             "during start-delay countdown, throttled to "
                             "1/3 frames during recording (small tiles).")
    parser.add_argument("--io-workers", type=int, default=16,
                        help="Async disk-write worker threads (default 16)")
    parser.add_argument("--queue-size", type=int, default=400,
                        help="Max pending write items (default 400)")
    parser.add_argument("--png-level", type=int, default=1,
                        help="PNG compression level 0-9 (default 1 = fast, "
                             "still lossless; default OpenCV is 3)")
    args = parser.parse_args()

    ctx = Context()

    if args.list:
        list_devices(ctx)
        return

    if not args.output:
        raise RuntimeError("--output is required (e.g., --output recorded_data/case_name)")

    master_serial = args.master_serial or MASTER_SERIAL
    if not master_serial:
        raise RuntimeError(
            "MASTER_SERIAL not set. Run --list first, then either edit "
            "the script or pass --master-serial <serial>."
        )

    # Output dirs
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    devices = ctx.query_devices()
    n = devices.get_count()
    if n < 2:
        raise RuntimeError("Need ≥2 cameras for multi-view recording")

    print(f"\nConfiguring sync (master={master_serial})...")
    ordered = configure_sync(devices, master_serial)

    # Per-cam output dirs
    for i in range(len(ordered)):
        os.makedirs(os.path.join(out_dir, "color", str(i)), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "depth", str(i)), exist_ok=True)

    # Build pipelines
    pipelines = []
    intrinsics = []
    serials = []
    for idx, (device, sn) in enumerate(ordered):
        is_master = (idx == 0)
        pipe = Pipeline(device)
        cfg = Config()

        # Color
        color_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_list.get_video_stream_profile(
                args.width, args.height, OBFormat.RGB, args.fps)
        except Exception as e:
            print(f"\n!!! Color {args.width}x{args.height} RGB @{args.fps}fps "
                  f"not supported on {sn}. Available profiles:")
            n_profiles = color_list.get_count()
            seen = set()
            for pi in range(n_profiles):
                prof = color_list.get_stream_profile_by_index(pi).as_video_stream_profile()
                fmt = prof.get_format()
                key = (prof.get_width(), prof.get_height(), int(fmt), prof.get_fps())
                if key in seen:
                    continue
                seen.add(key)
                print(f"    {prof.get_width()}x{prof.get_height()}  "
                      f"fmt={fmt}  fps={prof.get_fps()}")
            raise
        cfg.enable_stream(color_profile)

        # Depth — Gemini 2 uses Y14 format (uint16 buffer, top 2 bits 0);
        # AlignFilter will warp depth to color resolution.
        depth_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_list.get_video_stream_profile(
                args.depth_width, args.depth_height, OBFormat.Y14, args.fps)
        except Exception as e:
            print(f"\n!!! Depth {args.depth_width}x{args.depth_height} Y16 "
                  f"@{args.fps}fps not supported on {sn}. Available profiles:")
            n_profiles = depth_list.get_count()
            seen = set()
            for pi in range(n_profiles):
                prof = depth_list.get_stream_profile_by_index(pi).as_video_stream_profile()
                fmt = prof.get_format()
                key = (prof.get_width(), prof.get_height(), int(fmt), prof.get_fps())
                if key in seen:
                    continue
                seen.add(key)
                print(f"    {prof.get_width()}x{prof.get_height()}  "
                      f"fmt={fmt}  fps={prof.get_fps()}")
            raise
        cfg.enable_stream(depth_profile)

        # Read color intrinsics (depth will be aligned to color → use color K)
        intr = color_profile.get_intrinsic()
        intrinsics.append([
            [intr.fx, 0,       intr.cx],
            [0,       intr.fy, intr.cy],
            [0,       0,       1.0],
        ])
        serials.append(sn)

        # Align filter: warps depth onto color frame
        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        pipelines.append((pipe, cfg, align, is_master, sn))

    # Start SECONDARY first
    print("\nStarting SECONDARY pipelines...")
    for pipe, cfg, _, is_master, sn in pipelines:
        if not is_master:
            pipe.start(cfg)
            print(f"  slave {sn} started")
    time.sleep(0.5)

    # Then PRIMARY
    print("Starting PRIMARY pipeline...")
    for pipe, cfg, _, is_master, sn in pipelines:
        if is_master:
            pipe.start(cfg)
            print(f"  master {sn} started")

    # ---- PREVIEW HELPER ----
    n_cams = len(pipelines)
    tile_w, tile_h = 320, 180
    n_cols = min(n_cams, 3)
    n_rows = (n_cams + n_cols - 1) // n_cols

    def build_preview_grid(bgr_list, label_prefix=""):
        tiles = []
        for i, bgr in enumerate(bgr_list):
            if bgr is None:
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(tile, f"cam {i}: --", (8, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                tile = cv2.resize(bgr, (tile_w, tile_h))
                cv2.putText(tile, f"[{i}] {label_prefix}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tiles.append(tile)
        while len(tiles) < n_rows * n_cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows = [np.hstack(tiles[r * n_cols:(r + 1) * n_cols])
                for r in range(n_rows)]
        return np.vstack(rows)

    # ---- ASYNC WRITER POOL ----
    write_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_level]

    def writer_worker():
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break
            path, kind, arr = item
            try:
                if kind == "png":
                    cv2.imwrite(path, arr, png_params)
                else:  # npy
                    np.save(path, arr)
            except Exception as e:
                print(f"  [WARN writer] {path}: {e}", flush=True)
            write_queue.task_done()

    writers = [threading.Thread(target=writer_worker, daemon=True)
               for _ in range(args.io_workers)]
    for w in writers:
        w.start()

    # ---- PER-CAM CAPTURE THREADS ----
    # Phase 1: each thread warms up its own camera
    # Phase 2: idle-drain while main thread runs countdown (keeps SDK buffers clean)
    # Phase 3: record args.frames frames to queue
    warmup_barrier = threading.Barrier(n_cams + 1)
    go_event = threading.Event()
    stop_event = threading.Event()
    cam_progress = [0] * n_cams
    cam_warnings = [0] * n_cams
    cam_miss_color = [0] * n_cams
    cam_miss_depth = [0] * n_cams
    cam_miss_both = [0] * n_cams
    cam_latest_bgr: list = [None] * n_cams
    # Pure-capture timing (per-cam, excludes go_event latency + polling jitter)
    cam_t_first: list = [None] * n_cams
    cam_t_last: list = [None] * n_cams

    def capture_worker(cam_idx, pipe, align):
        # Phase 1: warmup
        for _ in range(args.warmup_frames):
            try:
                pipe.wait_for_frames(200)
            except Exception:
                pass
        warmup_barrier.wait()

        # Phase 2: idle-drain (keeps SDK buffer clean during countdown)
        while not go_event.is_set() and not stop_event.is_set():
            try:
                frames = pipe.wait_for_frames(200)
                if frames is not None and args.preview:
                    aligned = align.process(frames)
                    if aligned is not None:
                        cf = aligned.get_color_frame()
                        if cf is not None:
                            rgb = np.frombuffer(cf.get_data(), np.uint8).reshape(
                                args.height, args.width, 3)
                            cam_latest_bgr[cam_idx] = cv2.cvtColor(
                                rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        # Phase 3: record
        while cam_progress[cam_idx] < args.frames and not stop_event.is_set():
            f = cam_progress[cam_idx]
            try:
                frames = pipe.wait_for_frames(2000)
            except Exception as e:
                print(f"  [WARN] cam {cam_idx} frame {f}: {e}", flush=True)
                cam_warnings[cam_idx] += 1
                cam_progress[cam_idx] += 1
                continue
            if frames is None:
                cam_warnings[cam_idx] += 1
                cam_progress[cam_idx] += 1
                continue
            aligned = align.process(frames)
            if aligned is None:
                cam_warnings[cam_idx] += 1
                cam_progress[cam_idx] += 1
                continue
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if cf is None or df is None:
                cam_warnings[cam_idx] += 1
                if cf is None and df is None:
                    cam_miss_both[cam_idx] += 1
                elif cf is None:
                    cam_miss_color[cam_idx] += 1
                else:
                    cam_miss_depth[cam_idx] += 1
                cam_progress[cam_idx] += 1
                continue

            rgb = np.frombuffer(cf.get_data(), np.uint8).reshape(
                args.height, args.width, 3).copy()
            depth = np.frombuffer(df.get_data(), np.uint16).reshape(
                args.height, args.width).copy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Timestamp first/last successful capture (for pure-capture fps)
            now = time.time()
            if cam_t_first[cam_idx] is None:
                cam_t_first[cam_idx] = now
            cam_t_last[cam_idx] = now

            # The processing pipeline uses {frame_idx}.png (simple int, no padding)
            write_queue.put((
                os.path.join(out_dir, "color", str(cam_idx), f"{f}.png"),
                "png", bgr))
            write_queue.put((
                os.path.join(out_dir, "depth", str(cam_idx), f"{f}.npy"),
                "npy", depth))
            cam_latest_bgr[cam_idx] = bgr
            cam_progress[cam_idx] += 1

    capture_threads = [
        threading.Thread(target=capture_worker,
                         args=(i, pipe, align), daemon=True)
        for i, (pipe, _, align, _, _) in enumerate(pipelines)
    ]
    for t in capture_threads:
        t.start()

    # Wait for all cams to finish warmup
    print(f"\nWarming up ({args.warmup_frames} frames per cam, parallel)...")
    warmup_barrier.wait()
    print(f"Async I/O: {args.io_workers} writer threads, "
          f"PNG compression level {args.png_level}")
    print(f"Capture: {n_cams} parallel camera threads")

    # ---- START DELAY (walk-to-scene countdown) ----
    # Capture threads are idle-draining; main thread handles countdown + preview.
    if args.start_delay > 0:
        print(f"\nStarting in {args.start_delay:.1f}s (walk to scene)...")
        t_start = time.time()
        last_print_sec = None
        while True:
            elapsed = time.time() - t_start
            remaining = args.start_delay - elapsed
            if remaining <= 0:
                break
            cur_sec = int(np.ceil(remaining))
            if cur_sec != last_print_sec:
                print(f"  {cur_sec}...", flush=True)
                last_print_sec = cur_sec
            if args.preview:
                grid = build_preview_grid(
                    list(cam_latest_bgr), f"COUNTDOWN {cur_sec}s")
                cv2.imshow("record preview", grid)
                cv2.waitKey(1)
            else:
                time.sleep(0.05)
        print("  GO!\n")

    # ---- RECORD (triggered by go_event; capture threads do the work) ----
    print(f"Recording {args.frames} frames @ {args.fps} fps "
          f"({args.frames / args.fps:.1f}s)...")
    print("  Press Ctrl+C to abort early\n")

    t0 = time.time()
    go_event.set()
    last_report_bucket = -1
    last_preview_t = 0.0
    preview_interval = 1.0 / 30  # 30Hz preview — matches record rate
    try:
        while True:
            min_progress = min(cam_progress)
            if min_progress >= args.frames:
                break
            now = time.time()
            # Progress log every 30 frames (by slowest cam)
            bucket = min_progress // 30
            if bucket > last_report_bucket and min_progress > 0:
                elapsed = now - t0
                fps = min_progress / elapsed
                lag = max(cam_progress) - min(cam_progress)
                print(f"  frame {min_progress}/{args.frames}  "
                      f"({fps:.1f} fps actual, "
                      f"q={write_queue.qsize()}, cam lag={lag})")
                last_report_bucket = bucket
            # Preview
            if args.preview and (now - last_preview_t) > preview_interval:
                grid = build_preview_grid(
                    list(cam_latest_bgr),
                    f"REC {min_progress}/{args.frames}")
                cv2.imshow("record preview", grid)
                cv2.waitKey(1)
                last_preview_t = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[interrupted] waiting for capture threads to stop...")
        stop_event.set()

    # Join capture threads
    for t in capture_threads:
        t.join(timeout=3.0)

    # Flush writer queue
    print(f"Flushing write queue ({write_queue.qsize()} items)...")
    for _ in writers:
        write_queue.put(None)
    for w in writers:
        w.join()

    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    reached = min(cam_progress)
    print(f"\nRecorded {reached} frames in {elapsed:.2f}s "
          f"(wall-clock fps={reached / elapsed:.2f}, "
          f"incl. ~30ms start latency + poll jitter)")
    # Pure capture fps: between first and last successful frame on each cam
    # Measures actual hardware trigger rate, excludes go_event/polling overhead
    per_cam_fps = []
    for i in range(n_cams):
        if cam_t_first[i] is not None and cam_t_last[i] is not None \
                and cam_t_last[i] > cam_t_first[i] and cam_progress[i] > 1:
            dt = cam_t_last[i] - cam_t_first[i]
            per_cam_fps.append((cam_progress[i] - 1) / dt)
    if per_cam_fps:
        print(f"  Pure capture fps (per cam): "
              f"{[f'{v:.2f}' for v in per_cam_fps]} "
              f"(mean={sum(per_cam_fps)/len(per_cam_fps):.2f})")
    if any(cam_warnings):
        print(f"  Per-cam warnings: {cam_warnings}")
        print(f"    miss_color_only: {cam_miss_color}")
        print(f"    miss_depth_only: {cam_miss_depth}")
        print(f"    miss_both:       {cam_miss_both}")
    if max(cam_progress) != min(cam_progress):
        print(f"  Per-cam progress: {cam_progress} (mismatch indicates drops)")

    # ---- FILL DROPPED FRAMES (copy previous frame's file) ----
    # data_process pipeline expects contiguous 0.png / 0.npy ... (N-1).png /
    # (N-1).npy per cam. When a cam drops a frame, f_idx is advanced but no
    # file written → gap. Scan and fill by copying nearest prior frame.
    frame_num = min(cam_progress)
    fill_count = 0
    for cam_idx in range(n_cams):
        last_color = None
        last_depth = None
        for f in range(frame_num):
            color_p = os.path.join(out_dir, "color", str(cam_idx), f"{f}.png")
            depth_p = os.path.join(out_dir, "depth", str(cam_idx), f"{f}.npy")
            if os.path.exists(color_p):
                last_color = color_p
            elif last_color is not None:
                shutil.copy(last_color, color_p)
                fill_count += 1
            if os.path.exists(depth_p):
                last_depth = depth_p
            elif last_depth is not None:
                shutil.copy(last_depth, depth_p)
                fill_count += 1
    if fill_count > 0:
        print(f"  Filled {fill_count} missing files by copying prior frame "
              f"(nearest-neighbor interpolation)")

    # ---- GENERATE MP4 PER CAM ----
    # segment_util_video.py and dense_track.py both expect color/{i}.mp4
    # alongside color/{i}/*.png. Use ffmpeg if available, fall back to cv2.
    print(f"\nGenerating MP4 files (per cam)...")
    ffmpeg_available = shutil.which("ffmpeg") is not None
    for cam_idx in range(n_cams):
        color_dir = os.path.join(out_dir, "color", str(cam_idx))
        mp4_path = os.path.join(out_dir, "color", f"{cam_idx}.mp4")
        if ffmpeg_available:
            # Simple-int naming (0.png, 1.png, ...) → use concat demuxer via
            # shell pattern; ffmpeg -i %d.png handles non-padded integers.
            cmd = [
                "ffmpeg", "-y",
                "-r", str(args.fps),
                "-start_number", "0",
                "-i", os.path.join(color_dir, "%d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                mp4_path,
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"  cam {cam_idx}: {mp4_path} (H.264)")
            except subprocess.CalledProcessError as e:
                print(f"  [WARN] ffmpeg failed for cam {cam_idx}: {e}")
        else:
            # cv2 fallback: mp4v fourcc
            first_png = os.path.join(color_dir, "0.png")
            first = cv2.imread(first_png)
            if first is None:
                print(f"  [WARN] cam {cam_idx}: cannot read first frame")
                continue
            H, W = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(mp4_path, fourcc, args.fps, (W, H))
            for f in range(frame_num):
                img = cv2.imread(os.path.join(color_dir, f"{f}.png"))
                if img is not None:
                    vw.write(img)
            vw.release()
            print(f"  cam {cam_idx}: {mp4_path} (mp4v, install ffmpeg for H.264)")

    # Stop pipelines
    for pipe, _, _, _, _ in pipelines:
        pipe.stop()

    # ---- WRITE METADATA ----
    # frame_num uses slowest cam's progress (the common prefix all cams have)
    metadata = {
        "intrinsics": intrinsics,
        "WH": [args.width, args.height],
        "frame_num": min(cam_progress),
        "serial_numbers": serials,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata.json")

    # ---- COPY CALIBRATE.PKL ----
    if args.calibrate_pkl:
        if not os.path.exists(args.calibrate_pkl):
            print(f"  [WARN] calibrate.pkl not found at {args.calibrate_pkl}")
        else:
            shutil.copy(args.calibrate_pkl, os.path.join(out_dir, "calibrate.pkl"))
            print(f"Copied calibrate.pkl from {args.calibrate_pkl}")
    else:
        print(f"  [WARN] No --calibrate-pkl provided. Remember to copy it manually:")
        print(f"    cp <your_calibrate.pkl> {out_dir}/calibrate.pkl")

    print(f"\nDone. Output at: {out_dir}")
    print(f"\nNext steps:")
    print(f"  1. Copy {out_dir} to your DeformMaster data directory:")
    print(f"     rsync -avz {out_dir} \\")
    print(f"         <user>@<server>:/path/to/DeformMaster/data/different_types/")
    print(f"  2. Run the data_process pipeline from the repository root:")
    print(f"     python data_process/process_data.py --case_name {os.path.basename(out_dir)} \\")
    print(f"         --category <cloth|rope|softbody|package> --shape_prior")


if __name__ == "__main__":
    main()
