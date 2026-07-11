# Process to get the masks of the controller and the object
import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
parser.add_argument(
    "--cams",
    type=str,
    default="",
    help="Comma-separated camera ids (e.g. '0,2,4'). Empty = all available.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT

if args.cams.strip():
    cams = [int(c) for c in args.cams.split(",") if c.strip()]
else:
    cams = sorted(int(os.path.basename(p)) for p in glob.glob(f"{base_path}/{case_name}/depth/*")
                  if os.path.basename(p).isdigit())
assert len(cams) > 0, f"no camera depth dirs under {base_path}/{case_name}/depth"
print(f"Processing {case_name} on cams={cams}")

for camera_idx in cams:
    print(f"Processing {case_name} camera {camera_idx}")
    os.system(
        f"python ./data_process/segment_util_video.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT '{TEXT_PROMPT}' --camera_idx {camera_idx}"
    )
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
