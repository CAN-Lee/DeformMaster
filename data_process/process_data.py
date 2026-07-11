import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument(
    "--cams",
    type=str,
    default="",
    help="Comma-separated camera ids to use (e.g. '0,2,4'). "
         "Empty = auto-detect from data dirs (legacy behaviour, use all).",
)
parser.add_argument(
    "--no_regularize_tracks",
    action="store_true",
    default=False,
    help="Skip the 3D distance-preservation regularization step on "
         "object_points (default off — i.e. regularization runs).",
)
parser.add_argument(
    "--no_segment",
    action="store_true",
    default=False,
    help="Skip the GroundingDINO+SAM2 segmentation step. Use when masks were "
         "produced manually (e.g. via interactive_mask_app.py) and should not "
         "be overwritten. Requires mask/ and mask_info_<cam>.json to exist.",
)
parser.add_argument(
    "--controller_name",
    type=str,
    default="hand",
    help="Name of the controller/actuator that manipulates the object, used "
         "both as the GroundingDINO segmentation prompt ('{category}.{name}') "
         "and to pick the controller mask out of the detection results. "
         "Default 'hand'; use e.g. 'gripper' for robot end-effectors. Must be "
         "a word GroundingDINO returns verbatim as the mask label.",
)
args = parser.parse_args()
CAMS_FLAG = args.cams.strip()
CAMS_ARG = ["--cams", CAMS_FLAG] if CAMS_FLAG else []

# Set the debug flags
PROCESS_SEG = not args.no_segment
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_REGULARIZE = not args.no_regularize_tracks
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
CONTROLLER_NAME = args.controller_name
TEXT_PROMPT = f"{category}.{CONTROLLER_NAME}"
SHAPE_PRIOR = args.shape_prior

logger = None


def setup_logger(log_file="timer.log"):
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


_pipeline_failed = False


def run_step(cmd):
    """Run a shell command, check return code. Sets _pipeline_failed on error."""
    global _pipeline_failed
    if _pipeline_failed:
        print(f"[SKIP] {cmd[:80]}... (previous step failed)")
        return
    ret = os.system(cmd)
    if ret != 0:
        _pipeline_failed = True
        print(f"[ERROR] Command failed (exit {ret}): {cmd[:120]}")


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )



# ---- AUTO-FIX RECORDED DATA FORMAT ----
# record_multi_cam.py may produce zero-padded filenames (0000.png) or
# simple-int (0.png), and may or may not include per-cam .mp4 files.
# Normalize here so the rest of the pipeline always sees {f}.png + {i}.mp4.
import subprocess, shutil

case_dir = f"{base_path}/{case_name}"
meta_path = f"{case_dir}/metadata.json"
if os.path.exists(meta_path):
    with open(meta_path) as _mf:
        _meta = json.load(_mf)
    _num_cams = len(_meta["intrinsics"])
    _fps = _meta.get("fps", 30)

    for _ci in range(_num_cams):
        _color_dir = f"{case_dir}/color/{_ci}"
        _depth_dir = f"{case_dir}/depth/{_ci}"

        # 1) Rename zero-padded → simple-int (0000.png → 0.png)
        for _dir, _ext in [(_color_dir, ".png"), (_depth_dir, ".npy")]:
            if not os.path.isdir(_dir):
                continue
            _files = [f for f in os.listdir(_dir) if f.endswith(_ext)]
            if _files and _files[0][0] == '0' and len(_files[0].split('.')[0]) > 1:
                # Zero-padded detected
                for _f in _files:
                    _num = str(int(_f.replace(_ext, "")))
                    _new = _num + _ext
                    if _f != _new:
                        os.rename(os.path.join(_dir, _f),
                                  os.path.join(_dir, _new))

        # 2) Generate color/{i}.mp4 if missing
        _mp4 = f"{case_dir}/color/{_ci}.mp4"
        if not os.path.exists(_mp4) and os.path.isdir(_color_dir):
            # Conda env's ffmpeg may lack libx264; system ffmpeg has it
            _ffmpeg = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") \
                else shutil.which("ffmpeg")
            if os.path.isfile(_ffmpeg):
                subprocess.run([
                    _ffmpeg, "-y", "-r", str(_fps),
                    "-start_number", "0",
                    "-i", f"{_color_dir}/%d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-loglevel", "error", _mp4,
                ], check=False)
                if os.path.exists(_mp4):
                    print(f"[auto-fix] generated {_mp4}")

_CAMS_CLI = f" --cams {CAMS_FLAG}" if CAMS_FLAG else ""

if PROCESS_SEG:
    with Timer("Video Segmentation"):
        run_step(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT '{TEXT_PROMPT}'{_CAMS_CLI}"
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR and not _pipeline_failed:
    # Use the first selected camera (default cam 0) for the shape-prior anchor
    # frame, so '--cams' that omit cam 0 still pick a valid mask_info.
    if CAMS_FLAG:
        _shape_cam = int(CAMS_FLAG.split(",")[0].strip())
    else:
        _shape_cam = 0
    # Get the mask path for the image
    with open(f"{base_path}/{case_name}/mask/mask_info_{_shape_cam}.json", "r") as f:
        data = json.load(f)
    obj_ids = [int(key) for key, value in data.items() if value != CONTROLLER_NAME]
    if obj_ids:
        mask_path = f"{base_path}/{case_name}/mask/{_shape_cam}/{obj_ids[0]}/0.png"
    else:
        mask_path = ""
        print(f"[WARN] No object detected in cam {_shape_cam} mask_info, skipping shape prior")
        _pipeline_failed = True

    if not _pipeline_failed:
        existDir(f"{base_path}/{case_name}/shape")
        # Pre-placed shape prior shortcut: if shape/object.glb already exists
        # (e.g. dropped in by script_process_data.py from a curated mesh like
        # rope_mesh/best_rope.glb), skip the entire upscale + segment +
        # TRELLIS chain. Nothing downstream of object.glb consumes
        # high_resolution.png or masked_image.png.
        _shape_glb = f"{base_path}/{case_name}/shape/object.glb"
        if os.path.isfile(_shape_glb):
            print(f"[Shape Prior] {_shape_glb} already exists; "
                  f"skipping upscale + segment + TRELLIS")
        else:
            with Timer("Image Upscale"):
                if not os.path.isfile(f"{base_path}/{case_name}/shape/high_resolution.png"):
                    run_step(
                        f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/{_shape_cam}/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category '{category}'"
                    )

            with Timer("Image Segmentation"):
                run_step(
                    f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT '{category}' --output_path {base_path}/{case_name}/shape/masked_image.png"
                )

            with Timer("Shape Prior Generation"):
                run_step(
                    f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
                )

# Reset failure flag for non-shape-prior steps (shape prior failure shouldn't
# block track/pcd/mask which are independent)
if _pipeline_failed and SHAPE_PRIOR:
    print(f"[WARN] Shape prior chain failed, continuing with track/pcd pipeline")
    _pipeline_failed = False

if PROCESS_TRACK:
    with Timer("Dense Tracking"):
        run_step(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}{_CAMS_CLI}"
        )

if PROCESS_3D:
    with Timer("Lift to 3D"):
        run_step(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}{_CAMS_CLI}"
        )

    with Timer("Mask Post-Processing"):
        run_step(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name '{CONTROLLER_NAME}'{_CAMS_CLI}"
        )

    with Timer("Data Tracking"):
        run_step(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}{_CAMS_CLI}"
        )

    if PROCESS_REGULARIZE:
        with Timer("Track Regularization (3D distance preservation)"):
            run_step(
                f"python ./data_process/regularize_tracks.py --base_path {base_path} --case_name {case_name}"
            )

if PROCESS_ALIGN and SHAPE_PRIOR and not _pipeline_failed:
    with Timer("Alignment"):
        run_step(
            f"python ./data_process/align.py --base_path {base_path} --case_name {case_name} --controller_name '{CONTROLLER_NAME}'"
        )

if PROCESS_FINAL and not _pipeline_failed:
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            run_step(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )
        else:
            run_step(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)


# ---------------------------------------------------------------------------
# Cleanup intermediate per-frame pcd/*.npz to save disk (each frame ~24 MB,
# 150 frames ~3.6 GB per case). Safe once track_process_data.pkl exists,
# because mask + track post-processing have already consumed pcd/ into the
# pkl outputs and split.json frame_len has been written. Keep pcd/0.npz as
# a debug / quick-viz anchor for quick inspection.
# ---------------------------------------------------------------------------
_track_pkl = f"{base_path}/{case_name}/track_process_data.pkl"
_pcd_dir = f"{base_path}/{case_name}/pcd"
if os.path.isfile(_track_pkl) and os.path.isdir(_pcd_dir):
    _removed = 0
    for _f in os.listdir(_pcd_dir):
        if _f != "0.npz" and _f.endswith(".npz"):
            try:
                os.remove(os.path.join(_pcd_dir, _f))
                _removed += 1
            except OSError:
                pass
    if _removed:
        print(f"[cleanup] removed {_removed} intermediate pcd/*.npz "
              f"(kept pcd/0.npz)")
