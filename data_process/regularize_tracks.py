"""3D distance-preservation regularization for object_points tracks.

Reads ``<case>/track_process_data.pkl``, builds a kNN graph on the
frame-0 ``object_points``, records rest distances, then per-frame
projects the raw tracked positions to satisfy these distances via
Jacobi-style PBD iterations weighted against data attraction.

Rationale: CoTracker drifts on textureless / repetitive surfaces (rope
in particular). Approximate inextensibility — preserving 3D distances
between physically-close points — is a strong, cheap prior that pulls
drifted points back without a learned model. Distances are kept in
3D world frame because image-space distances are scale-distorted by
perspective and inconsistent across cams.

Backs up the original to ``track_process_data_raw.pkl`` (skipped if a
backup already exists, so re-running this step doesn't clobber it).

Usage::

    python data_process/regularize_tracks.py \\
        --base_path ./data/different_types --case_name my_double_compress_rope
"""
import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(p0, k, max_rest_factor):
    """Build a symmetric kNN graph on frame-0 positions.

    Returns (edges (E, 2) int64, rest_dists (E,) float32).

    Edges whose rest length exceeds ``max_rest_factor * median(rest)`` are
    dropped — those typically span across folds (e.g. a U-bent rope where
    spatial-NN sees a strand on the other side) and locking them in place
    would prevent the rope from straightening out.
    """
    n = p0.shape[0]
    tree = cKDTree(p0)
    dists, idxs = tree.query(p0, k=k + 1)        # +1 because self is included
    src = np.repeat(np.arange(n), k)
    dst = idxs[:, 1:].reshape(-1)
    d = dists[:, 1:].reshape(-1)

    # Symmetrize + dedup undirected edges.
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    keys = a.astype(np.int64) * n + b.astype(np.int64)
    _, uniq = np.unique(keys, return_index=True)
    src, dst, d = a[uniq], b[uniq], d[uniq]

    if max_rest_factor > 0 and len(d) > 0:
        med = float(np.median(d))
        keep = d <= max_rest_factor * med
        n_drop = int((~keep).sum())
        src, dst, d = src[keep], dst[keep], d[keep]
        if n_drop:
            print(f"[regularize] dropped {n_drop} edges with rest > "
                  f"{max_rest_factor:.1f}*median ({max_rest_factor * med:.4f}m)")

    edges = np.stack([src, dst], axis=1).astype(np.int64)
    return edges, d.astype(np.float32)


def regularize_frame(q_t, vis_t, edges, rest_d, num_iters, data_weight,
                     init):
    """Single-frame Jacobi PBD: alternate distance projection and data pull.

    q_t: (N, 3) raw tracked positions for this frame.
    vis_t: (N,) bool. Invisible points skip the data term — they ride the
        constraint network only (drift with their neighbors).
    init: (N, 3) starting estimate (warm-start from previous frame's
        regularized output gives temporal smoothness).
    """
    src, dst = edges[:, 0], edges[:, 1]
    n = q_t.shape[0]

    counts = np.zeros(n, dtype=np.float64)
    np.add.at(counts, src, 1.0)
    np.add.at(counts, dst, 1.0)
    counts = np.maximum(counts, 1.0)[:, None]

    p = init.astype(np.float64).copy()
    q = q_t.astype(np.float64)
    vis_w = (vis_t.astype(np.float64)[:, None] if vis_t is not None
             else np.ones((n, 1)))
    vis_w = vis_w * data_weight

    for _ in range(num_iters):
        # 1) Distance projection (Jacobi: accumulate + average by degree).
        delta = p[dst] - p[src]
        cur = np.linalg.norm(delta, axis=1) + 1e-9
        err = (cur - rest_d) / cur                          # (E,)
        corr = 0.5 * err[:, None] * delta                   # (E, 3)
        acc = np.zeros_like(p)
        np.add.at(acc, src, corr)
        np.add.at(acc, dst, -corr)
        p = p + acc / counts

        # 2) Data attraction (only for visible points).
        p = (1.0 - vis_w) * p + vis_w * q

    return p.astype(np.float32)


def regularize_tracks(track_data, k, max_rest_factor, num_iters, data_weight):
    obj = np.asarray(track_data["object_points"], dtype=np.float32)   # (T, N, 3)
    vis = np.asarray(track_data["object_visibilities"], dtype=bool)   # (T, N)
    t_total, n, _ = obj.shape

    edges, rest_d = build_knn_graph(obj[0], k=k, max_rest_factor=max_rest_factor)
    print(f"[regularize] kNN k={k}: {len(edges)} edges, "
          f"rest median={np.median(rest_d):.4f}m, max={rest_d.max():.4f}m")

    out = obj.copy()
    err_before, err_after = [], []
    for t in range(1, t_total):
        d_pre = np.linalg.norm(obj[t, edges[:, 1]] - obj[t, edges[:, 0]], axis=1)
        err_before.append(float(np.abs(d_pre - rest_d).mean()))
        out[t] = regularize_frame(
            obj[t], vis[t], edges, rest_d,
            num_iters=num_iters, data_weight=data_weight,
            init=out[t - 1],
        )
        d_post = np.linalg.norm(out[t, edges[:, 1]] - out[t, edges[:, 0]], axis=1)
        err_after.append(float(np.abs(d_post - rest_d).mean()))

    disp = float(np.mean(np.linalg.norm(out - obj, axis=2)))
    print(f"[regularize] edge-length |err|: "
          f"before={np.mean(err_before):.4f}m -> after={np.mean(err_after):.4f}m  "
          f"(rest median {np.median(rest_d):.4f}m)")
    print(f"[regularize] mean per-point displacement: {disp:.4f}m")

    track_data["object_points"] = out
    return track_data


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--case_name", required=True)
    ap.add_argument("--k", type=int, default=8,
                    help="kNN neighbors per vertex on frame 0.")
    ap.add_argument("--max-rest-factor", type=float, default=3.0,
                    help="Drop edges whose frame-0 rest length exceeds "
                         "FACTOR * median rest length (cross-fold filter). "
                         "0 disables.")
    ap.add_argument("--iters", type=int, default=15,
                    help="Jacobi PBD iterations per frame.")
    ap.add_argument("--data-weight", type=float, default=0.3,
                    help="Per-iteration weight pulling toward the raw "
                         "tracked position (0..1). Higher = trust tracks "
                         "more, less smoothing.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Don't write track_process_data_raw.pkl backup.")
    args = ap.parse_args()

    case_dir = Path(args.base_path) / args.case_name
    pkl = case_dir / "track_process_data.pkl"
    backup = case_dir / "track_process_data_raw.pkl"

    if not pkl.is_file():
        raise SystemExit(f"[regularize] missing {pkl}")

    # First-time backup only — re-running shouldn't clobber the raw copy.
    if not args.no_backup and not backup.is_file():
        shutil.copyfile(pkl, backup)
        print(f"[regularize] backup -> {backup}")

    with open(pkl, "rb") as f:
        track_data = pickle.load(f)

    track_data = regularize_tracks(
        track_data,
        k=args.k,
        max_rest_factor=args.max_rest_factor,
        num_iters=args.iters,
        data_weight=args.data_weight,
    )

    with open(pkl, "wb") as f:
        pickle.dump(track_data, f)
    print(f"[regularize] wrote {pkl}")


if __name__ == "__main__":
    main()
