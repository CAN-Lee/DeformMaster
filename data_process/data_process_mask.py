# Process the mask data to filter out the outliers and generate the processed masks

import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
import os
import glob
import cv2
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
parser.add_argument(
    "--cams",
    type=str,
    default="",
    help="Comma-separated camera ids (e.g. '0,2,4'). Empty = all available.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name

processed_masks = {}


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, cams):
    """``cams`` is the list of cam ids; ``points``/``masks``/``colors`` arrays
    are indexed by *position* in ``cams``, but per-cam files (mask PNGs and
    mask_info entries) are indexed by ``cam_id`` directly."""
    global processed_masks
    processed_masks[frame_idx] = {}

    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for pos, cam_id in enumerate(cams):
        # Load the object mask (merge multiple detections via OR)
        mask = np.zeros_like(masks[pos])
        for oid in mask_info[cam_id]["object"]:
            mask = np.logical_or(mask, read_mask(f"{mask_path}/{cam_id}/{oid}/{frame_idx}.png"))
        object_mask = np.logical_and(masks[pos], mask)
        object_points = points[pos][object_mask]
        object_colors = colors[pos][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        controller_mask = np.zeros_like(masks[pos])
        for controller_idx in mask_info[cam_id]["controller"]:
            mask = read_mask(f"{mask_path}/{cam_id}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[pos], controller_mask)
        controller_points = points[pos][controller_mask]
        controller_colors = colors[pos][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply the outlier removal (radius + DBSCAN)
    # Step 1: radius outlier — removes isolated noise
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    # Step 2: DBSCAN — removes clustered noise (keep clusters >= threshold)
    # "Keep largest only" would delete valid object parts when the object is
    # split into 2 spatial clusters (e.g., rope occluded by hand in the middle).
    min_object_cluster_pts = 100
    if len(object_pcd.points) > 0:
        labels = np.array(object_pcd.cluster_dbscan(eps=0.02, min_points=10))
        if len(labels) > 0 and (labels >= 0).any():
            counts = np.bincount(labels[labels >= 0])
            small_clusters = set(np.where(counts < min_object_cluster_pts)[0])
            noise = labels == -1
            small = np.array([l in small_clusters for l in labels])
            remove_idx = np.where(noise | small)[0]
            if len(remove_idx) > 0:
                dbscan_removed = np.asarray(
                    object_pcd.select_by_index(remove_idx).points
                )
                filtered_object_points = np.concatenate(
                    [filtered_object_points, dbscan_removed]
                )
                object_pcd = object_pcd.select_by_index(remove_idx, invert=True)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    # DBSCAN for controller: keep ALL clusters with >= min_cluster_pts points
    # (double_* has 2 hands = 2 large clusters; "keep largest only" would
    # delete one hand)
    min_cluster_pts = 50
    if len(controller_pcd.points) > 0:
        labels = np.array(controller_pcd.cluster_dbscan(eps=0.02, min_points=10))
        if len(labels) > 0 and (labels >= 0).any():
            counts = np.bincount(labels[labels >= 0])
            small_clusters = set(np.where(counts < min_cluster_pts)[0])
            noise = labels == -1
            small = np.array([l in small_clusters for l in labels])
            remove_idx = np.where(noise | small)[0]
            if len(remove_idx) > 0:
                dbscan_removed = np.asarray(
                    controller_pcd.select_by_index(remove_idx).points
                )
                filtered_controller_points = np.concatenate(
                    [filtered_controller_points, dbscan_removed]
                )
                controller_pcd = controller_pcd.select_by_index(
                    remove_idx, invert=True
                )

    # Convert filtered points to sets for O(1) lookup (was O(N*M) linear scan)
    _filtered_obj_set = set(map(tuple, filtered_object_points))
    _filtered_ctrl_set = set(map(tuple, filtered_controller_points))

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for pos, cam_id in enumerate(cams):
        # processed_masks key uses *position* (matches points[pos] indexing
        # downstream in data_process_track.py)
        processed_masks[frame_idx][pos] = {}
        # Load the object mask (merge multiple detections via OR)
        mask = np.zeros_like(masks[pos])
        for oid in mask_info[cam_id]["object"]:
            mask = np.logical_or(mask, read_mask(f"{mask_path}/{cam_id}/{oid}/{frame_idx}.png"))
        object_mask = np.logical_and(masks[pos], mask)
        object_points = points[pos][object_mask]
        indices = np.nonzero(object_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered (removed) set
        object_indices = [
            j for j, point in enumerate(object_points)
            if tuple(point) in _filtered_obj_set
        ]
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][pos]["object"] = object_mask

        # Load the controller mask
        controller_mask = np.zeros_like(masks[pos])
        for controller_idx in mask_info[cam_id]["controller"]:
            mask = read_mask(f"{mask_path}/{cam_id}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[pos], controller_mask)
        controller_points = points[pos][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered (removed) set
        controller_indices = [
            j for j, point in enumerate(controller_points)
            if tuple(point) in _filtered_ctrl_set
        ]
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][pos]["controller"] = controller_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[pos][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[pos][object_mask])

        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[pos][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[pos][controller_mask])

        controller_pcd += pcd

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])

    return object_pcd, controller_pcd


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"

    if args.cams.strip():
        cams = [int(c) for c in args.cams.split(",") if c.strip()]
    else:
        cams = sorted(int(os.path.basename(p).split("_")[-1].split(".")[0])
                      for p in glob.glob(f"{mask_path}/mask_info_*.json"))
    assert len(cams) > 0, f"no mask_info_*.json under {mask_path}"
    print(f"[mask] using cams={cams}")
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))
    # Load the mask metadata, keyed by cam_id (not position)
    mask_info = {}
    for cam_id in cams:
        with open(f"{base_path}/{case_name}/mask/mask_info_{cam_id}.json", "r") as f:
            data = json.load(f)
        mask_info[cam_id] = {"object": [], "controller": []}
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                mask_info[cam_id]["object"].append(int(key))
            else:
                mask_info[cam_id]["controller"].append(int(key))

    VIZ = os.environ.get("DATA_PROCESS_VIZ", "0") == "1"

    if VIZ:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, cams
        )
        if VIZ:
            if i == 0:
                object_pcd = temp_object_pcd
                controller_pcd = temp_controller_pcd
                vis.add_geometry(object_pcd)
                vis.add_geometry(controller_pcd)
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
                object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
                controller_pcd.points = o3d.utility.Vector3dVector(
                    temp_controller_pcd.points
                )
                controller_pcd.colors = o3d.utility.Vector3dVector(
                    temp_controller_pcd.colors
                )
                vis.update_geometry(object_pcd)
                vis.update_geometry(controller_pcd)
                vis.poll_events()
                vis.update_renderer()

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)

    # Deprecated for now
    # # Generate the videos with for masked objects and controllers
    # exist_dir(f"{base_path}/{case_name}/temp_mask")
    # for i in range(num_cam):
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/object")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/controller")
    #     object_idx = mask_info[i]["object"]
    #     for frame_idx in range(frame_num):
    #         object_mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
    #         img = cv2.imread(f"{base_path}/{case_name}/color/{i}/{frame_idx}.png")
    #         masked_object_img = cv2.bitwise_and(
    #             img, img, mask=object_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/object/{frame_idx}.png",
    #             masked_object_img,
    #         )

    #         controller_mask = np.zeros_like(object_mask)
    #         for controller_idx in mask_info[i]["controller"]:
    #             mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
    #             controller_mask = np.logical_or(controller_mask, mask)
    #         masked_controller_img = cv2.bitwise_and(
    #             img, img, mask=controller_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/controller/{frame_idx}.png",
    #             masked_controller_img,
    #         )

    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/object/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/object_{i}.mp4"
    #     )
    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/controller/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/controller_{i}.mp4"
    #     )
