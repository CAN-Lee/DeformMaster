# `gaussian_splatting/` - Upstream License Notice

The files under this directory are derived from third-party projects and are
not covered by the top-level MIT `LICENSE`. Each file preserves its original
copyright header. The principal upstream source is:

- **3D Gaussian Splatting** by GRAPHDECO, Inria
  https://github.com/graphdeco-inria/gaussian-splatting
  Released for non-commercial, research and evaluation use only. See the
  upstream `LICENSE.md` for the full terms.

Additional third-party headers found in this directory include:

- The PlenOctree Authors (https://github.com/sxyu/plenoctree)
- ETH Zurich & UNC Chapel Hill (COLMAP-derived utilities)

If you need a permissively licensed rasterizer/renderer, consider replacing
this directory with [`gsplat`](https://github.com/nerfstudio-project/gsplat)
(Apache-2.0); the API surface used in `interactive_playground_online.py` and
`deformmaster/render/` is small enough that the swap is feasible.

By using code in this directory you agree to comply with the upstream licenses
listed above.
