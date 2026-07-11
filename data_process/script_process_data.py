"""Parallel multi-GPU data processing for multiple cases.

Each case gets its own GPU via CUDA_VISIBLE_DEVICES. Cases are pulled
from a shared queue so GPUs stay saturated even if cases finish at
different times (same pattern as script_stage3_train.py).

Usage:
    python data_process/script_process_data.py --config data_config_c3.csv --gpus 0,1,2,3,4
    python data_process/script_process_data.py --config configs/data_process/data_config.csv               # serial (default)

The script auto-relaunches itself under ``xvfb-run -a`` on headless
servers (when ``DISPLAY`` is unset), because Open3D / PyVista in the
data_process pipeline init OpenGL at import time and crash without an X
display. Set ``NO_XVFB=1`` to disable, or pre-set ``DISPLAY`` to skip.
"""
import os
import shutil
import sys

# --- auto xvfb-run on headless servers ---
if (os.environ.get("DISPLAY") is None
        and not os.environ.get("NO_XVFB")):
    _xvfb = shutil.which("xvfb-run")
    if _xvfb is not None:
        os.execv(_xvfb, [_xvfb, "-a", sys.executable] + sys.argv)
    else:
        print("[script_process_data] WARN: no DISPLAY and xvfb-run not found; "
              "Open3D / PyVista may crash. Install xvfb-run or set DISPLAY.",
              file=sys.stderr)

import argparse
import csv
import multiprocessing as mp
import signal
import subprocess
import time

PYTHON = sys.executable

# Raw capture inputs — never deleted by --overwrite. Anything else inside
# the case dir is treated as a process_data.py output and wiped.
# c2w_sequence.npy is a raw input for monocular cases (per-frame camera poses
# from extract_mono_video.py); data_process_pcd.py needs it to cancel
# camera motion, so it must survive overwrite just like calibrate.pkl.
_RAW_INPUTS_KEEP = frozenset(
    ("calibrate.pkl", "metadata.json", "color", "depth", "c2w_sequence.npy"))

# Default per-category curated shape priors. If a case's case_name OR category
# contains the keyword (case-insensitive substring), the GLB is pre-placed at
# <case>/shape/object.glb after the overwrite-clean step, and process_data.py
# skips the TRELLIS upscale+segment+shape_prior chain for that case.
DEFAULT_SHAPE_PRIOR_PRESETS = (
    "rope:data/mesh/best_rope.glb",
    "bear:data/mesh/best_bear.glb",
    "rabbit:data/mesh/best_rabbit.glb",
    "teddy:data/mesh/best_teddy.glb",
)


def parse_shape_prior_presets(specs):
    """Parse a list of 'KEYWORD:/abs/path.glb' strings into [(kw_lower, path)]."""
    out = []
    for s in specs:
        if ":" not in s:
            print(f"[shape_prior_preset] WARN: bad spec {s!r}, expected "
                  f"'KEYWORD:/abs/path.glb', skipping", flush=True)
            continue
        kw, p = s.split(":", 1)
        kw = kw.strip().lower()
        p = p.strip()
        if not kw or not p:
            print(f"[shape_prior_preset] WARN: empty key or path in {s!r}, skipping",
                  flush=True)
            continue
        out.append((kw, p))
    return out


def match_preset(case_name, category, presets):
    """Return the GLB path if any preset's keyword matches this case (and the
    file exists); else None."""
    name = (case_name or "").lower()
    cat = (category or "").lower()
    for kw, path in presets:
        if kw in name or kw in cat:
            if os.path.isfile(path):
                return path
            print(f"[shape_prior_preset] WARN: preset for {kw!r} matched "
                  f"{case_name} but glb missing at {path}; falling back to TRELLIS",
                  flush=True)
            return None
    return None


def clean_case_outputs(base_path, case_name, keep_mask=False):
    """Wipe everything inside the case dir except the raw capture inputs.

    keep_mask=True additionally preserves mask/ (manual masks from
    interactive_mask_app.py), so a --no-segment rerun still regenerates
    cotracker/pcd/shape/etc. without clobbering the hand-authored masks."""
    case_dir = os.path.join(base_path, case_name)
    if not os.path.isdir(case_dir):
        return []
    keep = _RAW_INPUTS_KEEP | {"mask"} if keep_mask else _RAW_INPUTS_KEEP
    removed = []
    for name in os.listdir(case_dir):
        if name in keep:
            continue
        path = os.path.join(case_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
            removed.append(name + "/")
        else:
            os.remove(path)
            removed.append(name)
    return removed

# Global list for cleanup on Ctrl+C
_active_workers = []
_active_subprocesses = []


def worker(gpu_id, task_queue, base_path, cams, controller_name, no_segment):
    gpu_id = str(gpu_id).strip()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    while True:
        # Blocking get + None sentinel: avoids the mp.Queue feeder race
        # where get_nowait() can raise Empty before put() items become
        # visible to the child process (caused some GPUs to silently
        # exit before any task was assigned).
        task = task_queue.get()
        if task is None:
            break
        case_name, category, shape_prior, regularize = task

        cmd = [
            PYTHON, "data_process/process_data.py",
            "--base_path", base_path,
            "--case_name", case_name,
            "--category", category,
        ]
        if cams:
            cmd.extend(["--cams", cams])
        cmd.extend(["--controller_name", controller_name])
        if no_segment:
            cmd.append("--no_segment")
        if shape_prior.lower() == "true":
            cmd.append("--shape_prior")
        if regularize.lower() != "true":
            cmd.append("--no_regularize_tracks")

        print(f"[GPU {gpu_id}] >>> {case_name}", flush=True)
        try:
            proc = subprocess.Popen(cmd, env=env)
            _active_subprocesses.append(proc)
            proc.wait()
            _active_subprocesses.remove(proc)
            if proc.returncode == 0:
                print(f"[GPU {gpu_id}] <<< {case_name} OK", flush=True)
            else:
                print(f"[GPU {gpu_id}] !!! {case_name} FAILED", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] !!! {case_name} ERROR: {e}", flush=True)


def _check_case_outputs(base_path, case_name, cams, shape_prior,
                        preset_used=False):
    """Return list of (label, ok, detail) tuples for one case.

    ``preset_used``: True if this case used a curated shape-prior GLB; the
    TRELLIS chain is skipped, so high_resolution.png / masked_image.png are
    not expected to exist and are omitted from the checks."""
    case_dir = os.path.join(base_path, case_name)
    checks = []

    # ---- per-cam segmentation ----
    n_mask_dirs = 0
    for c in cams:
        d = os.path.join(case_dir, "mask", str(c))
        if os.path.isdir(d) and os.listdir(d):
            n_mask_dirs += 1
    checks.append(("mask/<cam>/<obj>/*.png", n_mask_dirs == len(cams),
                   f"{n_mask_dirs}/{len(cams)} cams"))

    n_mask_info = sum(
        os.path.isfile(os.path.join(case_dir, "mask", f"mask_info_{c}.json"))
        for c in cams
    )
    checks.append(("mask/mask_info_<cam>.json", n_mask_info == len(cams),
                   f"{n_mask_info}/{len(cams)} cams"))

    # ---- per-cam dense tracking ----
    n_track = sum(
        os.path.isfile(os.path.join(case_dir, "cotracker", f"{c}.npz"))
        for c in cams
    )
    checks.append(("cotracker/<cam>.npz", n_track == len(cams),
                   f"{n_track}/{len(cams)} cams"))

    # ---- aggregate (post per-cam) ----
    pcd_dir = os.path.join(case_dir, "pcd")
    pcd_n = (len([f for f in os.listdir(pcd_dir) if f.endswith(".npz")])
             if os.path.isdir(pcd_dir) else 0)
    checks.append(("pcd/<frame>.npz", pcd_n > 0, f"{pcd_n} frames"))

    checks.append(("mask/processed_masks.pkl",
                   os.path.isfile(os.path.join(case_dir, "mask", "processed_masks.pkl")),
                   ""))
    checks.append(("track_process_data.pkl",
                   os.path.isfile(os.path.join(case_dir, "track_process_data.pkl")),
                   ""))

    # ---- shape prior chain (only if requested) ----
    if shape_prior:
        # Skip the upscale/segment intermediates when a curated GLB was
        # pre-placed (process_data.py also skips generating them).
        chain = (
            ("shape/object.glb", "(preset)" if preset_used else "(TRELLIS)"),
            ("shape/matching/final_mesh.glb", ""),  # align.py output
        )
        if not preset_used:
            chain = (
                ("shape/high_resolution.png", ""),
                ("shape/masked_image.png", ""),
            ) + chain
        for fname, detail in chain:
            checks.append((fname,
                           os.path.isfile(os.path.join(case_dir, fname)),
                           detail))

    # ---- final outputs ----
    checks.append(("final_data.pkl",
                   os.path.isfile(os.path.join(case_dir, "final_data.pkl")), ""))
    checks.append(("split.json",
                   os.path.isfile(os.path.join(case_dir, "split.json")), ""))
    return checks


def print_checklist(tasks, base_path, cams_arg, config_path, preset_used=None):
    """Per-case post-run checklist of expected pipeline outputs.

    Tees output to stdout AND a text file at
    ``<base_path>/checklist_<config_basename>.txt``.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f"  Pipeline Output Checklist  ({len(tasks)} case(s))")
    lines.append("=" * 72)

    cams_global = ([int(c) for c in cams_arg.split(",") if c.strip()]
                   if cams_arg else None)
    fully_done = 0
    for case_name, _, shape_prior, _ in tasks:
        sp_bool = (shape_prior.lower() == "true")
        if cams_global is not None:
            cams_eff = cams_global
        else:
            depth_dir = os.path.join(base_path, case_name, "depth")
            cams_eff = (sorted(int(d) for d in os.listdir(depth_dir) if d.isdigit())
                        if os.path.isdir(depth_dir) else [])

        used_glb = (preset_used or {}).get(case_name)
        rows = _check_case_outputs(base_path, case_name, cams_eff, sp_bool,
                                   preset_used=bool(used_glb))
        total = len(rows)
        passed_ok = sum(1 for _, ok, _ in rows if ok)
        head = ("[OK]  " if passed_ok == total else f"[{passed_ok}/{total}]")
        lines.append("")
        sp_tag = (f"shape_prior=preset({os.path.basename(used_glb)})"
                  if used_glb else f"shape_prior={sp_bool}")
        lines.append(f"{head} {case_name}  cams={cams_eff}  {sp_tag}")
        for label, ok, detail in rows:
            mark = " OK " if ok else "MISS"
            extra = f"  ({detail})" if detail else ""
            lines.append(f"  [{mark}] {label:32s}{extra}")
        if passed_ok == total:
            fully_done += 1

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Summary: {fully_done}/{len(tasks)} case(s) fully complete")
    lines.append("=" * 72)

    text = "\n".join(lines) + "\n"
    print("\n" + text, flush=True)

    # Tee to file under base_path (one level above each case dir).
    config_basename = os.path.splitext(os.path.basename(config_path))[0]
    out_path = os.path.join(base_path, f"checklist_{config_basename}.txt")
    try:
        os.makedirs(base_path, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[checklist] wrote {out_path}", flush=True)
    except Exception as e:
        print(f"[checklist] WARN: failed to write {out_path}: {e}", flush=True)


def cleanup_and_exit(signum, frame):
    """Kill all child processes on Ctrl+C."""
    print("\n[Ctrl+C] Killing all workers and subprocesses...", flush=True)
    # Kill subprocesses (process_data.py instances)
    for proc in _active_subprocesses:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    # Kill worker processes
    for w in _active_workers:
        try:
            w.kill()
        except Exception:
            pass
    # Kill any remaining children in our process group
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:
        pass
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data_process/data_config.csv",
                        help="CSV: case_name,category,shape_prior_bool (no header)")
    parser.add_argument("--base-path", default="./data/different_types")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU ids for parallel processing. "
                             "Pass --gpus '' for serial mode.")
    parser.add_argument("--cams", default="0,1,2,3,4",
                        help="Comma-separated camera ids to process. "
                             "Pass --cams '' to auto-detect from data dirs (use all).")
    parser.add_argument("--controller-name", default="hand",
                        help="Controller/actuator that manipulates the object, "
                             "passed through to process_data.py. Default 'hand'; "
                             "use 'gripper' for robot end-effectors. Applies to "
                             "all cases in the run.")
    parser.add_argument("--no-segment", action="store_true", default=False,
                        help="Skip GroundingDINO+SAM2 segmentation for every "
                             "case (masks authored manually via "
                             "interactive_mask_app.py). The overwrite-clean step "
                             "automatically preserves mask/ in this mode while "
                             "still regenerating cotracker/pcd/shape/etc.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Before processing, wipe cotracker/mask/pcd/shape/"
                             "split.json/track_process_data.pkl for each case in "
                             "the CSV (raw inputs untouched). Pass --no-overwrite "
                             "to keep existing outputs.")
    parser.add_argument("--stagger-seconds", type=float, default=15.0,
                        help="Seconds to wait between launching successive "
                             "GPU workers (parallel mode), to avoid "
                             "simultaneous CPU/RAM spikes from concurrent "
                             "model loading (SAM2 / Grounding-DINO / CoTracker). "
                             "Set 0 to disable.")
    parser.add_argument("--shape-prior-preset", action="append", default=None,
                        help="Curated shape-prior GLB to skip TRELLIS for "
                             "matching cases. Format 'KEYWORD:/abs/path.glb'. "
                             "Match rule: KEYWORD (case-insensitive substring) "
                             "in case_name OR category. Pre-placed at "
                             "<case>/shape/object.glb after the overwrite-clean "
                             "step. Repeatable. Default: "
                             f"{', '.join(DEFAULT_SHAPE_PRIOR_PRESETS)}.")
    args = parser.parse_args()

    base_path = args.base_path

    # Parse tasks from CSV
    tasks = []
    with open(args.config, newline="", encoding="utf-8") as csvfile:
        for row in csv.reader(csvfile):
            if not row or row[0].startswith("#"):
                continue
            case_name, category, shape_prior = row[0], row[1], row[2]
            # 4th column = regularize_tracks ('True'/'False'). Legacy 3-column
            # rows default to 'True' (regularize ON, matches the process_data
            # default).
            regularize = row[3] if len(row) > 3 else "True"
            if not os.path.exists(f"{base_path}/{case_name}"):
                print(f"[SKIP] {case_name}: data dir not found")
                continue
            tasks.append((case_name, category, shape_prior, regularize))

    if not tasks:
        print("No tasks to process.")
        return

    if args.overwrite:
        print(f"[overwrite] Wiping processed outputs for {len(tasks)} case(s) "
              f"(raw inputs kept).", flush=True)
        if args.no_segment:
            print("[overwrite] --no-segment: preserving mask/ (manual masks)", flush=True)
        for case_name, _, _, _ in tasks:
            removed = clean_case_outputs(base_path, case_name, keep_mask=args.no_segment)
            if removed:
                print(f"[overwrite] {case_name}: removed {' '.join(removed)}", flush=True)

    # Resolve shape-prior presets and pre-place curated GLBs for matched
    # cases (must run AFTER overwrite-clean which would otherwise wipe them).
    preset_specs = (args.shape_prior_preset
                    if args.shape_prior_preset is not None
                    else list(DEFAULT_SHAPE_PRIOR_PRESETS))
    presets = parse_shape_prior_presets(preset_specs)
    preset_used = {}  # case_name -> glb path (for checklist)
    if presets:
        print(f"[shape_prior_preset] presets: "
              f"{[f'{k}:{p}' for k, p in presets]}", flush=True)
        for case_name, category, shape_prior, _ in tasks:
            if shape_prior.lower() != "true":
                continue
            glb = match_preset(case_name, category, presets)
            if not glb:
                continue
            shape_dir = os.path.join(base_path, case_name, "shape")
            os.makedirs(shape_dir, exist_ok=True)
            dst = os.path.join(shape_dir, "object.glb")
            shutil.copyfile(glb, dst)
            preset_used[case_name] = glb
            print(f"[shape_prior_preset] {case_name}: shape/object.glb <- {glb}",
                  flush=True)

    os.system("rm -f timer.log")

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    cams = args.cams.strip() if args.cams else ""
    controller_name = (args.controller_name or "hand").strip() or "hand"
    no_segment = args.no_segment

    if args.gpus and args.gpus.strip():
        # ---- PARALLEL MODE ----
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        print(f"[parallel] {len(tasks)} cases on GPUs={gpus}, cams={cams or '(auto)'}, "
              f"controller='{controller_name}'")

        task_queue = mp.Queue()
        for t in tasks:
            task_queue.put(t)
        # one None sentinel per worker so each worker's blocking get()
        # has a clean termination signal once tasks run out
        for _ in range(len(gpus)):
            task_queue.put(None)

        for i, gpu_id in enumerate(gpus):
            if i > 0 and args.stagger_seconds > 0:
                # Stagger worker spawns so SAM2 / Grounding-DINO / CoTracker
                # model loads don't all happen at the same instant, avoiding
                # transient CPU/RAM spikes that can OOM-kill the host.
                time.sleep(args.stagger_seconds)
            p = mp.Process(target=worker,
                           args=(gpu_id, task_queue, base_path, cams, controller_name, no_segment))
            p.start()
            _active_workers.append(p)
            print(f"[parallel] launched worker for GPU {gpu_id} "
                  f"({i + 1}/{len(gpus)})", flush=True)
        for p in _active_workers:
            p.join()

        print(f"[parallel] All done.")
    else:
        # ---- SERIAL MODE (original behavior) ----
        for case_name, category, shape_prior, regularize in tasks:
            cmd = [
                PYTHON, "data_process/process_data.py",
                "--base_path", base_path,
                "--case_name", case_name,
                "--category", category,
            ]
            if cams:
                cmd.extend(["--cams", cams])
            cmd.extend(["--controller_name", controller_name])
            if no_segment:
                cmd.append("--no_segment")
            if shape_prior.lower() == "true":
                cmd.append("--shape_prior")
            if regularize.lower() != "true":
                cmd.append("--no_regularize_tracks")
            print(f"[serial] >>> {case_name}", flush=True)
            rc = subprocess.call(cmd)
            if rc == 0:
                print(f"[serial] <<< {case_name} OK", flush=True)
            else:
                print(f"[serial] !!! {case_name} FAILED", flush=True)

    # Per-case post-run checklist of expected pipeline outputs.
    print_checklist(tasks, base_path, cams, args.config,
                    preset_used=preset_used)


if __name__ == "__main__":
    main()
