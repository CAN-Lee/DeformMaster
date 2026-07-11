"""Interactive point-prompt masking for DeformMaster cases (single cam).

A Gradio app that replaces the automatic GroundingDINO step in
``segment_util_video.py`` when detection fails (thin/occluded objects, words
GroundingDINO doesn't know, etc.). You click positive/negative points on a
frame, SAM2 (image predictor) previews the mask live, you name the object,
add as many objects as you want, then SAM2 video predictor propagates each
object from *its own annotation frame* (bidirectionally) over the whole clip.

Output matches the contract consumed by ``data_process_mask.py``:
  <case>/mask/mask_info_<cam>.json   {obj_id(str): label}
  <case>/mask/<cam>/<obj_id>/<frame>.png   binary mask (255/0)

The controller object's label must equal the ``--controller_name`` you later
pass to the pipeline (e.g. 'black gripper'); every other label is treated as
object. After saving, rerun the pipeline with ``--no_segment`` so this mask is
not overwritten.

Usage:
    python data_process/interactive_mask_app.py \
        --base_path ./data/different_types --case_name <case_name> \
        --camera_idx 0 --port 8890
Then forward the port and open the URL.
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

import gradio as gr

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CKPT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _frame_index(p):
    return int(os.path.splitext(os.path.basename(p))[0])


class MaskSession:
    """Holds SAM2 models + per-object annotations for one case/camera."""

    def __init__(self, base_path, case_name, camera_idx):
        self.base_path = base_path
        self.case_name = case_name
        self.cam = camera_idx
        self.case_dir = os.path.join(base_path, case_name)
        self.color_dir = os.path.join(self.case_dir, "color", str(camera_idx))

        self.frame_paths = sorted(
            (os.path.join(self.color_dir, f) for f in os.listdir(self.color_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))),
            key=_frame_index,
        )
        if not self.frame_paths:
            raise RuntimeError(f"No frames found under {self.color_dir}")
        self.num_frames = len(self.frame_paths)

        # SAM2 video predictor needs a jpg frame dir named so that sorted order
        # == frame index. Build it once under a temp dir; index i -> {i:05d}.jpg
        self.jpg_dir = tempfile.mkdtemp(prefix=f"sam2_{case_name}_{camera_idx}_")
        for i, p in enumerate(self.frame_paths):
            img = Image.open(p).convert("RGB")
            img.save(os.path.join(self.jpg_dir, f"{i:05d}.jpg"), quality=95)
        self.frame_hw = np.array(Image.open(self.frame_paths[0])).shape[:2]

        print(f"[mask_app] loading SAM2 ({DEVICE})...")
        self.video_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
        self.image_predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE))

        # interaction state for the object currently being annotated
        self.cur_frame = 0
        self.cur_points = []     # list of [x, y]
        self.cur_labels = []     # list of 1 (pos) / 0 (neg)
        self.cur_pos = True      # currently adding positive points
        self._image_set_for = None

        # committed objects: list of dicts {ann_frame, points, labels, name}
        self.objects = []

    # ---- frame / image ----
    def load_frame(self, i):
        i = int(i)
        self.cur_frame = i
        self.cur_points, self.cur_labels = [], []
        self._image_set_for = None
        return np.array(Image.open(self.frame_paths[i]).convert("RGB"))

    def _ensure_image(self, img):
        if self._image_set_for != self.cur_frame:
            self.image_predictor.set_image(img)
            self._image_set_for = self.cur_frame

    # ---- point prompting (live preview via image predictor) ----
    def add_point(self, img, x, y):
        self._ensure_image(img)
        self.cur_points.append([x, y])
        self.cur_labels.append(1 if self.cur_pos else 0)
        return self._preview(img)

    def _preview(self, img):
        if not self.cur_points:
            return img
        masks, scores, _ = self.image_predictor.predict(
            point_coords=np.array(self.cur_points, dtype=np.float32),
            point_labels=np.array(self.cur_labels, dtype=np.int32),
            multimask_output=True,
        )
        mask = masks[int(np.argmax(scores))].astype(bool)
        return self._overlay(img, mask)

    def _overlay(self, img, mask):
        out = img.copy()
        out[mask] = (0.45 * out[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        for (x, y), lb in zip(self.cur_points, self.cur_labels):
            c = (0, 255, 0) if lb == 1 else (255, 0, 0)
            x, y = int(x), int(y)
            out[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = c
        return out


def build_demo(sess: MaskSession):
    with gr.Blocks() as demo:
        gr.Markdown(
            f"### Interactive mask — `{sess.case_name}` cam {sess.cam} "
            f"({sess.num_frames} frames)\n"
            "Click points on the frame to carve out one object, name it, then "
            "**Commit object**. Repeat for every object (the manipulator's "
            "label must match your `--controller_name`). Finally **Propagate** "
            "and **Save**."
        )
        status = gr.Textbox(label="Status", interactive=False,
                            value="Pick a frame, click positive points.")
        with gr.Row():
            with gr.Column():
                frame_slider = gr.Slider(0, sess.num_frames - 1, value=0, step=1,
                                         label="Annotation frame")
                frame_img = gr.Image(sess.load_frame(0), label="Frame (click to add points)",
                                     type="numpy")
                with gr.Row():
                    pos_btn = gr.Button("Positive points", min_width=40)
                    neg_btn = gr.Button("Negative points", min_width=40)
                    clear_btn = gr.Button("Clear points", min_width=40)
            with gr.Column():
                preview_img = gr.Image(label="Mask preview", type="numpy")
                obj_name = gr.Textbox(label="Object label",
                                      placeholder="e.g. 'twine' or 'black gripper'")
                commit_btn = gr.Button("Commit object ✓")
                committed = gr.Dataframe(headers=["obj_id", "label", "ann_frame", "#pts"],
                                         label="Committed objects", interactive=False)
                propagate_btn = gr.Button("Propagate to all frames ▶")
                preview_vid = gr.Video(label="Tracked preview")
                save_btn = gr.Button("Save masks to case 💾")

        def _committed_table():
            return [[i, o["name"], o["ann_frame"], len(o["points"])]
                    for i, o in enumerate(sess.objects)]

        def on_frame(i):
            return sess.load_frame(i), "Loaded frame %d. Click points." % int(i)

        def on_pos():
            sess.cur_pos = True
            return "Adding POSITIVE points (object interior)."

        def on_neg():
            sess.cur_pos = False
            return "Adding NEGATIVE points (exclude region)."

        def on_clear(img):
            sess.cur_points, sess.cur_labels = [], []
            return img, "Cleared points for current object."

        def on_click(img, evt: gr.SelectData):
            x, y = evt.index[0], evt.index[1]
            return sess.add_point(img, x, y)

        def on_commit(name):
            if not sess.cur_points:
                return _committed_table(), "No points yet — click some first."
            if not name or not name.strip():
                return _committed_table(), "Give the object a label first."
            sess.objects.append({
                "ann_frame": sess.cur_frame,
                "points": list(sess.cur_points),
                "labels": list(sess.cur_labels),
                "name": name.strip(),
            })
            sess.cur_points, sess.cur_labels = [], []
            return (_committed_table(),
                    f"Committed '{name.strip()}' (#{len(sess.objects) - 1}). "
                    "Add another object or Propagate.")

        def on_propagate():
            if not sess.objects:
                return None, "Commit at least one object first."
            vp = sess.video_predictor
            state = vp.init_state(video_path=sess.jpg_dir)
            vp.reset_state(state)
            ann_frames = set()
            for oid, o in enumerate(sess.objects):
                vp.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=o["ann_frame"],
                    obj_id=oid,
                    points=np.array(o["points"], dtype=np.float32),
                    labels=np.array(o["labels"], dtype=np.int32),
                )
                ann_frames.add(o["ann_frame"])

            # Propagate forward and backward so objects annotated mid-clip cover
            # the whole sequence.
            segs = {}  # frame -> {oid: bool mask}

            def collect(reverse):
                start = max(ann_frames) if reverse else min(ann_frames)
                for f_idx, obj_ids, logits in vp.propagate_in_video(
                        state, start_frame_idx=start, reverse=reverse):
                    d = segs.setdefault(f_idx, {})
                    for k, oid in enumerate(obj_ids):
                        d[oid] = (logits[k] > 0.0).cpu().numpy().squeeze()

            collect(reverse=False)
            collect(reverse=True)
            sess._segs = segs

            # build an overlay preview video
            import imageio.v2 as iio
            out_path = os.path.join(sess.jpg_dir, "tracked_preview.mp4")
            palette = [(0, 255, 0), (255, 80, 0), (60, 120, 255), (255, 255, 0)]
            writer = iio.get_writer(out_path, fps=15, macro_block_size=1)
            for i in range(sess.num_frames):
                base = np.array(Image.open(sess.frame_paths[i]).convert("RGB"))
                for oid, m in segs.get(i, {}).items():
                    c = palette[oid % len(palette)]
                    base[m] = (0.5 * base[m] + 0.5 * np.array(c)).astype(np.uint8)
                writer.append_data(base)
            writer.close()
            return out_path, f"Propagated {len(sess.objects)} object(s) over {sess.num_frames} frames."

        def on_save():
            if not getattr(sess, "_segs", None):
                return "Propagate first, then Save."
            mask_root = os.path.join(sess.case_dir, "mask")
            cam_dir = os.path.join(mask_root, str(sess.cam))
            # wipe any prior masks for this cam so stale obj_ids don't linger
            if os.path.isdir(cam_dir):
                shutil.rmtree(cam_dir)
            os.makedirs(cam_dir, exist_ok=True)
            info = {}
            for oid, o in enumerate(sess.objects):
                info[str(oid)] = o["name"]
                od = os.path.join(cam_dir, str(oid))
                os.makedirs(od, exist_ok=True)
                for i in range(sess.num_frames):
                    m = sess._segs.get(i, {}).get(oid)
                    if m is None:
                        m = np.zeros(sess.frame_hw, dtype=bool)
                    Image.fromarray((m.astype(np.uint8) * 255)).save(
                        os.path.join(od, f"{i}.png"))
            with open(os.path.join(mask_root, f"mask_info_{sess.cam}.json"), "w") as f:
                json.dump(info, f)
            return (f"Saved {len(sess.objects)} object(s) to {cam_dir} and "
                    f"mask_info_{sess.cam}.json. Now rerun the pipeline with "
                    "--no_segment.")

        frame_slider.change(on_frame, [frame_slider], [frame_img, status])
        frame_img.select(on_click, [frame_img], [preview_img])
        pos_btn.click(on_pos, outputs=[status])
        neg_btn.click(on_neg, outputs=[status])
        clear_btn.click(on_clear, [frame_img], [preview_img, status])
        commit_btn.click(on_commit, [obj_name], [committed, status])
        propagate_btn.click(on_propagate, outputs=[preview_vid, status])
        save_btn.click(on_save, outputs=[status])

    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_path", default="./data/different_types")
    ap.add_argument("--case_name", required=True)
    ap.add_argument("--camera_idx", type=int, default=0)
    ap.add_argument("--port", type=int, default=8890)
    args = ap.parse_args()

    sess = MaskSession(args.base_path, args.case_name, args.camera_idx)
    demo = build_demo(sess)
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                allowed_paths=[os.path.abspath(sess.case_dir), sess.jpg_dir])


if __name__ == "__main__":
    main()
