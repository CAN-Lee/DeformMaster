import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
import json
from argparse import ArgumentParser

"""
Hyperparam for Ground and Tracking
"""

# Put below base path into args
parser = ArgumentParser()
parser.add_argument("--base_path", type=str, default="data/real_collect")
parser.add_argument("--case_name", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
parser.add_argument("--camera_idx", type=int)
parser.add_argument("--output_path", type=str, default="NONE")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_idx = args.camera_idx
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"
else:
    output_path = args.output_path

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"
existDir(f"{base_path}/{case_name}/tmp_data")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}")

SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"

"""
Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
"""
# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)


# init sam image predictor and video predictor model
sam2_checkpoint = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image_source, image = load_image(img_path)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
confidences = confidences.numpy().tolist()
class_names = labels

print(input_boxes)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(image_source)

# process the detection results
OBJECTS = class_names

print(OBJECTS)

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"

if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
else:
    raise NotImplementedError(
        "SAM 2 video predictor only support point/box/mask prompts"
    )

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for (
    out_frame_idx,
    out_obj_ids,
    out_mask_logits,
) in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

existDir(f"{output_path}/mask/")
existDir(f"{output_path}/mask/{camera_idx}")

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

# Save the id_to_objects into json
with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
    json.dump(ID_TO_OBJECTS, f)

for frame_idx, masks in video_segments.items():
    for obj_id, mask in masks.items():
        existDir(f"{output_path}/mask/{camera_idx}/{obj_id}")
        # mask is 1 * H * W
        Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
            f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
        )
