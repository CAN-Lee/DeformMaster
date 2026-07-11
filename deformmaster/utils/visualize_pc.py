import os


def _is_headless():
    if os.name == "posix":
        display = os.environ.get("DISPLAY")
        return not display or not display.strip()
    return False


if _is_headless():
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")

import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import open3d as o3d
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _create_video_from_frames(frame_paths, output_path, fps, cleanup=True):
    if not frame_paths:
        raise RuntimeError("No frames captured; cannot create video.")

    frame_dir = os.path.dirname(frame_paths[0])
    ffmpeg_binary = None
    codec = "libx264"
    for candidate in ("/usr/bin/ffmpeg", "ffmpeg"):
        try:
            result = subprocess.run(
                [candidate, "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            continue
        if "libx264" in result.stdout:
            ffmpeg_binary = candidate
            break

    if ffmpeg_binary is None:
        ffmpeg_binary = "ffmpeg"
        codec = "libopenh264"

    cmd = [
        ffmpeg_binary,
        "-y",
        "-r",
        str(fps),
        "-i",
        os.path.join(frame_dir, "frame_%06d.png"),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if cleanup:
            for path in frame_paths:
                if os.path.exists(path):
                    os.remove(path)


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    *,
    intrinsics,
    w2cs,
    image_size,
    overlay_path=None,
    fps=30,
    visualize=False,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    """Render a point-cloud trajectory using explicit camera parameters.

    This is the small subset of the legacy visualizer needed by
    inference_mpm.py. It intentionally avoids global config state.
    """
    if save_video and not save_path:
        raise ValueError("save_path is required when save_video=True")
    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    width, height = image_size
    intrinsic = np.asarray(intrinsics[vis_cam_idx])
    w2c = np.asarray(w2cs[vis_cam_idx])

    object_points = _to_numpy(object_points)
    object_colors = _to_numpy(object_colors)
    controller_points = _to_numpy(controller_points)
    object_visibilities = _to_numpy(object_visibilities)

    if object_colors is None:
        object_colors = np.tile([1.0, 0.0, 0.0], (object_points.shape[0], object_points.shape[1], 1))
    elif object_colors.shape[1] < object_points.shape[1]:
        pad = np.full(
            (object_colors.shape[0], object_points.shape[1] - object_colors.shape[1], 3),
            0.3,
            dtype=object_colors.dtype,
        )
        object_colors = np.concatenate([object_colors, pad], axis=1)

    headless = _is_headless()
    if headless and visualize and not save_video:
        print("WARNING: headless environment detected; skipping interactive visualization.")
        return

    vis = o3d.visualization.Visualizer()
    window_created = False
    try:
        vis.create_window(visible=visualize and not headless, width=width, height=height)
        window_created = True
    except Exception as exc:
        if not save_video:
            print(f"WARNING: cannot create Open3D window; skipping visualization: {exc}")
            return
        try:
            vis.create_window(visible=False, width=width, height=height)
            window_created = True
        except Exception as exc2:
            print(f"WARNING: cannot create offscreen Open3D window; skipping video render: {exc2}")
            return

    temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_") if save_video else None
    frame_paths = []
    controller_meshes = []
    prev_center = []

    try:
        for i in range(object_points.shape[0]):
            object_pcd = o3d.geometry.PointCloud()
            if object_visibilities is None:
                object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
                object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
            else:
                visible_idx = np.where(object_visibilities[i])[0]
                object_pcd.points = o3d.utility.Vector3dVector(object_points[i, visible_idx, :])
                object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i, visible_idx, :])

            if i == 0:
                render_object_pcd = object_pcd
                vis.add_geometry(render_object_pcd)
                if controller_points is not None:
                    for j in range(controller_points.shape[1]):
                        origin = controller_points[i, j]
                        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(origin)
                        mesh.compute_vertex_normals()
                        mesh.paint_uniform_color([1, 0, 0])
                        controller_meshes.append(mesh)
                        prev_center.append(origin.copy())
                        vis.add_geometry(mesh)

                view_control = vis.get_view_control()
                if view_control is not None:
                    camera_params = o3d.camera.PinholeCameraParameters()
                    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic)
                    camera_params.extrinsic = w2c
                    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
            else:
                render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
                render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
                vis.update_geometry(render_object_pcd)
                if controller_points is not None:
                    for j in range(controller_points.shape[1]):
                        origin = controller_points[i, j]
                        controller_meshes[j].translate(origin - prev_center[j])
                        vis.update_geometry(controller_meshes[j])
                        prev_center[j] = origin.copy()

            vis.poll_events()
            vis.update_renderer()

            if save_video:
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frame = (frame * 255).astype(np.uint8)
                if overlay_path is not None:
                    image_path = os.path.join(overlay_path, str(vis_cam_idx), f"{i}.png")
                    if os.path.exists(image_path):
                        overlay = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                        mask = np.all(frame == [255, 255, 255], axis=-1)
                        frame[mask] = overlay[mask]

                frame_path = os.path.join(temp_frame_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)

            if visualize:
                time.sleep(1 / fps)

        if save_video:
            _create_video_from_frames(frame_paths, save_path, fps, cleanup=True)
    finally:
        if window_created:
            vis.destroy_window()
        if temp_frame_dir and os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
