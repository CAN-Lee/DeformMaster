# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument(
    "--cams",
    type=str,
    default="",
    help="Comma-separated camera ids (e.g. '0,2,4'). Empty = use all (range(num_cam)).",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


# Adapted from Open3D RGB-D camera helper utilities.
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


def getPcdFromDepth(depth, intrinsic):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic).T
    points = points.reshape(H, W, 3)
    return points


def get_pcd_from_data(path, frame_idx, cams, intrinsics, c2ws):
    """``cams`` is a list of cam ids (e.g. [0,2,4]); per-cam files are read by
    cam_id, intrinsics/c2ws indexed by cam_id, but the returned aggregate
    arrays are indexed by *position* in ``cams`` (length = len(cams))."""
    total_points = []
    total_colors = []
    total_masks = []
    for cam_id in cams:
        color = cv2.imread(f"{path}/color/{cam_id}/{frame_idx}.png")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{cam_id}/{frame_idx}.npy") / 1000.0

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[cam_id],
        )
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 1.5)
        points_flat = points.reshape(-1, 3)
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[cam_id], homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)
    # pcd = o3d.geometry.PointCloud()
    # visualize_points = []
    # visualize_colors = []
    # for i in range(num_cam):
    #     visualize_points.append(
    #         total_points[i][total_masks[i]].reshape(-1, 3)
    #     )
    #     visualize_colors.append(
    #         total_colors[i][total_masks[i]].reshape(-1, 3)
    #     )
    # visualize_points = np.concatenate(visualize_points)
    # visualize_colors = np.concatenate(visualize_colors)
    # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
    # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
    # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
    # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
    # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
    # visualize_points = visualize_points[mask]
    # visualize_colors = visualize_colors[mask]
        
    # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])
    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    print(data["serial_numbers"])

    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    if args.cams.strip():
        cams = [int(c) for c in args.cams.split(",") if c.strip()]
    else:
        cams = list(range(num_cam))
    assert len(cams) > 0, "no cams selected"
    for c in cams:
        assert 0 <= c < num_cam, f"cam id {c} out of range (have {num_cam} cams in metadata)"
    print(f"[pcd] num_cam={num_cam}, using cams={cams}")

    # For monocular extractions the camera can drift/jitter across frames,
    # but calibrate.pkl only stores frame-0's c2w. Projecting every frame's
    # depth with that single pose alias-es the camera motion into the world
    # frame (static background appears to jitter, cloth picks up a bogus
    # drift that later poisons MPM training). If a full per-frame pose
    # sequence was dumped alongside (via extract_mono_video.py),
    # use it so each frame is projected with its own camera pose.
    c2w_sequence = None
    c2w_seq_path = f"{base_path}/{case_name}/c2w_sequence.npy"
    if num_cam == 1 and len(cams) == 1 and cams[0] == 0 and os.path.exists(c2w_seq_path):
        c2w_sequence = np.load(c2w_seq_path)
        assert c2w_sequence.shape[0] >= frame_num, (
            f"c2w_sequence.npy has {c2w_sequence.shape[0]} poses but "
            f"metadata frame_num is {frame_num}"
        )
        print(
            f"[mono] Using per-frame c2w from c2w_sequence.npy "
            f"({c2w_sequence.shape[0]} poses)"
        )

    exist_dir(f"{base_path}/{case_name}/pcd")

    VIZ = os.environ.get("DATA_PROCESS_VIZ", "0") == "1"

    if VIZ:
        cameras = []
        for i in cams:
            camera = getCamera(
                c2ws[i],
                intrinsics[i, 0, 0],
                intrinsics[i, 1, 1],
                intrinsics[i, 0, 2],
                intrinsics[i, 1, 2],
                z_flip=True,
                scale=0.2,
            )
            cameras += camera

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for camera in cameras:
            vis.add_geometry(camera)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coordinate)

    pcd = None
    for i in tqdm(range(frame_num)):
        if c2w_sequence is not None:
            # mono special case: rebuild a 1-element c2ws-like list, indexed by cam_id 0
            c2ws_for_frame = [c2w_sequence[i]]
        else:
            c2ws_for_frame = c2ws
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, cams, intrinsics, c2ws_for_frame
        )

        if VIZ:
            if i == 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.add_geometry(pcd)
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )
