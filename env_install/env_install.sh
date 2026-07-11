#!/bin/bash
set -euo pipefail

SKIP_OPT="${SKIP_OPT:-0}"

conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install taichi omegaconf plyfile tqdm h5py
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

if [ "$SKIP_OPT" != "1" ]; then
    # Optional data-processing dependencies: RealSense, Grounded-SAM,
    # Grounding-DINO, SDXL upscaling, and TRELLIS shape priors.
    pip install Cython
    pip install pyrealsense2
    pip install atomics
    pip install pynput

    pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git

    pip install diffusers
    pip install accelerate

    cd data_process
    if [ ! -d TRELLIS ]; then
        git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
    fi
    cd TRELLIS
    . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    cd ../..
else
    echo "Skipping optional data-processing dependencies."
fi

pip install gsplat==1.4.0
pip install kornia
cd gaussian_splatting/
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
cd ..
