"""Shape prior generation with TRELLIS.

The script can re-exec itself under the Python configured through
``DEFORMMASTER_TRELLIS_PYTHON`` before importing the pipeline.

Usage::
    python data_process/shape_prior.py \
        --img_path <case>/shape/masked_image.png \
        --output_dir <case>/shape
Output: <output_dir>/object.glb
"""
import os
import sys

# --------- self-dispatch ---------------------------------------------------
TRELLIS_PY = os.environ.get("DEFORMMASTER_TRELLIS_PYTHON", sys.executable)

# Force the env vars trellis 1 needs BEFORE either dispatch or in-process import.
# Setting only before execv would miss the case where the user invokes this
# script directly with the trellis1 env's python (no dispatch needed).
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

target_py = TRELLIS_PY
if os.path.realpath(sys.executable) != os.path.realpath(target_py):
    if os.path.isfile(target_py):
        os.execv(target_py, [target_py, os.path.abspath(__file__)] + sys.argv[1:])
    else:
        print(f"[shape_prior] TRELLIS Python not found at {target_py}. "
              "Set DEFORMMASTER_TRELLIS_PYTHON=/path/to/python or run inside the "
              "matching TRELLIS environment.")
        sys.exit(1)

# --- from here on we're in the right trellis env ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from argparse import ArgumentParser
from PIL import Image


def _run_trellis(args):
    """TRELLIS (Microsoft, structured 3D latents, ~1B params)."""
    # Resolve user-facing paths to absolute BEFORE we chdir; otherwise the
    # subsequent chdir into TRELLIS_REPO breaks relative paths like
    # "./data/different_types/<case>/shape/masked_image.png".
    args.img_path = os.path.abspath(args.img_path)
    args.output_dir = os.path.abspath(args.output_dir)

    # Repo is vendored at data_process/TRELLIS/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    TRELLIS1_REPO = os.path.abspath(os.path.join(_HERE, "TRELLIS"))
    sys.path.insert(0, TRELLIS1_REPO)
    # TRELLIS 1's from_pretrained resolves nested ckpt paths (e.g.
    # "ckpts/ss_flow_img_dit_L_16l8_fp16") relative to cwd, so we must
    # chdir into the repo before invoking it.
    os.chdir(TRELLIS1_REPO)

    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    print(f"[shape_prior] Loading TRELLIS ({args.model})...")
    pipe = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()

    print(f"[shape_prior] Image: {args.img_path}")
    image = Image.open(args.img_path)

    print(f"[shape_prior] Running inference...")
    out = pipe.run(image, seed=args.seed)
    mesh = out["mesh"][0]
    gaussian = out["gaussian"][0]
    print(f"[shape_prior] Raw mesh: V={len(mesh.vertices)} F={len(mesh.faces)}")

    print(f"[shape_prior] Exporting GLB (simplify={args.simplify}, texture_size={args.texture_size})...")
    glb = postprocessing_utils.to_glb(
        gaussian, mesh,
        simplify=args.simplify,
        texture_size=args.texture_size,
        verbose=False,
    )
    out_path = os.path.join(args.output_dir, "object.glb")
    glb.export(out_path)
    print(f"[shape_prior] Saved {out_path}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str,
                        default="JeffreyXiang/TRELLIS-image-large",
                        help="Hugging Face model ID for TRELLIS.")
    parser.add_argument("--simplify", type=float, default=0.95,
                        help="Face decimation ratio (0.95 keeps 5%).")
    parser.add_argument("--texture-size", type=int, default=1024,
                        help="Texture resolution (4096 OOMs pytorch3d).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible mesh generation.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _run_trellis(args)


if __name__ == "__main__":
    main()
