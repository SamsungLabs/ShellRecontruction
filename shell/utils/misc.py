"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""
# im2mesh/utils/misc.py

import math
import numpy_indexed as npi


import torch
import scipy
import numpy as np
import pyvista as pv
import open3d as o3d
from typing import List, Tuple



def depth_image_to_pointcloud3d(
    depth_image: np.ndarray,
    camera_k: np.ndarray,
    mask: np.ndarray = None,
    subsample: int = 1,
):
    """Convert a depth image (HWC) to a pointcloud.

    Args:
        depth_image: Depth as an array (HWC). Units should be consistent with camera_k
        camera_k: Camera intrinsics. Must be the same units as the depth image, so likely mm
        [optional] mask: NxMx1 mask to apply to the depth image before extracting a pointcloud.
        [optional] subsample: Factor by which to subsample points.

    Returns:
        PointCloud3D of the projected depth image realtive to the camera frame in the same units
        as camera_k and the depth image
    """
    v_size, u_size, _ = depth_image.shape

    # Extract lists of coordinates.
    u_img_range = np.arange(0, u_size)
    v_img_range = np.arange(0, v_size)
    u_grid, v_grid = np.meshgrid(u_img_range, v_img_range)
    u_img, v_img, d = (
        u_grid.ravel(),
        v_grid.ravel(),
        depth_image[v_grid, u_grid].ravel(),
    )

    # Apply mask.
    if mask is not None:
        v_grid, u_grid = np.where(mask.squeeze())
        u_img, v_img, d = (
            u_grid.ravel(),
            v_grid.ravel(),
            depth_image[v_grid, u_grid].ravel(),
        )

    # Subsample, potentially.
    if subsample != 1:
        u_img, v_img, d = u_img[::subsample], v_img[::subsample], d[::subsample]

    # Convert to camera frame.
    pc = depth_coords_to_camera_points(
        np.stack((u_img, v_img)), np.expand_dims(d, axis=0), camera_k
    )

    # ignore points that projected to 0
    d_zero_mask = pc[2, :] != 0
    pc = pc[:, d_zero_mask]

    return pc


def depth_coords_to_camera_points(
    coords: np.ndarray, z_vals: np.ndarray, camera_k: np.ndarray
):
    """Transform a set of depth-valued coordinates in the image frame into a PointCloud.

    Args:
        coords: Coordinates in the form [u, v]
        z_vals: Depth values.
        camera_k: Intrinsics.
        frame: the frame enum for the camera.

    Returns:
        A PointCloud representing the depth image projected into the camera frame.
    """
    assert coords.shape[0] == 2 and z_vals.shape[0] == 1, "{} {}".format(
        coords.shape[0], z_vals.shape[0]
    )
    assert coords.shape[1] == z_vals.shape[1]

    # Invert K.
    k_inv = np.linalg.inv(camera_k)

    # Homogenize the coordinats in form [u, v, 1].
    homogenous_uvs = np.concatenate((coords, np.ones((1, coords.shape[1]))))

    # Get the unscaled position for each of the points in the image frame.
    unscaled_points = k_inv @ homogenous_uvs

    # Scale the points by their depth values.
    scaled_points = np.multiply(unscaled_points, z_vals)

    # return PointCloud3D(scaled_points)
    return scaled_points


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_pcd(obj):
    if isinstance(obj, pv.PolyData):
        pcd = obj
    elif obj.pcd is None:
        assert obj.mesh is not None
        pcd = pv.PolyData(obj.mesh.points)
    else:
        pcd = obj.pcd
    return pcd


def interpolate(img, res):
    assert img.shape[0] == img.shape[1]
    img_resc = img
    if img.shape[0] == res:
        return img
    if img.size != 16 and img.size != 8:
        factor = img.shape[0] / res

        img_resc = scipy.ndimage.zoom(
            img, [1 / factor, 1 / factor] + [1] * (len(img.shape) - 2), order=1
        )
    return img_resc


def persp2ortho(
    persp_depth,
    px_size,
    camera_k=None,
    view_angle=90,
    min_z_step=1e-2,
    upsample_factor=4,
):
    return perspective2orthogonal(
        persp_depth, px_size, camera_k, view_angle, min_z_step, upsample_factor
    )


def perspective2orthogonal(
    persp_depth,
    px_size,
    camera_k=None,
    view_angle=90,
    min_z_step=1e-2,
    upsample_factor=4,
):
    original_res = persp_depth.shape[-1]
    persp_depth = persp_depth.copy()
    persp_depth[persp_depth == 0] = np.nan
    persp_depth_ups = interpolate(persp_depth, original_res * upsample_factor)
    persp_depth_ups[persp_depth_ups != persp_depth_ups] = 0

    if camera_k is None:
        camera_k = get_camera_k_from_angle(original_res, view_angle)
    camera_k = adjust_camera_k_by_res(
        camera_k, original_res=original_res, new_res=original_res * upsample_factor
    )

    depth_zeroed = persp_depth_ups.copy()
    depth_zeroed[depth_zeroed != depth_zeroed] = 0
    points = get_visible_pcd(depth_zeroed, camera_k=camera_k, return_type="np")

    z = points[:, -1]
    xy = points[:, :-1]

    xy_quant = np.rint(xy / px_size).astype(np.int32) + original_res // 2
    # good_point_ids = (xy_quant[:, 0] < original_res) * (xy_quant[:, 1] < original_res)

    # xy_quant = xy_quant[good_point_ids]
    # z = z[good_point_ids]

    point_ids_by_z = np.argsort(-z)

    orth_farthest_point = np.zeros([original_res, original_res])
    orth_depth_sum = np.zeros([1, original_res, original_res])
    orth_depth_layer_count = np.zeros([original_res, original_res], dtype=np.int32)
    orth_depth_add_count = np.zeros([1, original_res, original_res], dtype=np.int32)

    for point_id in point_ids_by_z:
        point_uv = xy_quant[point_id]
        point_z = z[point_id]
        assert point_z != 0
        if point_uv[0] >= original_res or point_uv[1] >= original_res:
            continue
        else:
            if orth_depth_layer_count[point_uv[0], point_uv[1]] == 0:
                orth_depth_sum[0, point_uv[0], point_uv[1]] = point_z

                orth_depth_add_count[0, point_uv[0], point_uv[1]] = 1
                orth_depth_layer_count[point_uv[0], point_uv[1]] = 1

            else:
                prev_point_layer_count = (
                    orth_depth_layer_count[point_uv[0], point_uv[1]] - 1
                )
                assert orth_farthest_point[point_uv[0], point_uv[1]] >= point_z

                if orth_farthest_point[point_uv[0], point_uv[1]] - point_z > min_z_step:
                    new_point_layer_count = prev_point_layer_count + 1
                    if new_point_layer_count >= orth_depth_sum.shape[0]:
                        orth_depth_sum = np.concatenate(
                            [orth_depth_sum, np.zeros([1, original_res, original_res])]
                        )
                        orth_depth_add_count = np.concatenate(
                            [
                                orth_depth_add_count,
                                np.zeros(
                                    [1, original_res, original_res], dtype=np.int32
                                ),
                            ]
                        )
                    orth_depth_sum[
                        new_point_layer_count, point_uv[0], point_uv[1]
                    ] += point_z
                    orth_depth_layer_count[point_uv[0], point_uv[1]] += 1
                    orth_depth_add_count[
                        new_point_layer_count, point_uv[0], point_uv[1]
                    ] += 1
                else:
                    orth_depth_sum[
                        prev_point_layer_count, point_uv[0], point_uv[1]
                    ] += point_z
                    orth_depth_add_count[
                        prev_point_layer_count, point_uv[0], point_uv[1]
                    ] += 1

            orth_farthest_point[point_uv[0], point_uv[1]] = point_z

    orth_depth = orth_depth_sum
    orth_depth[orth_depth != 0] /= orth_depth_add_count[orth_depth != 0]

    orth_depth = clean_up_visible_orth_depth(orth_depth)
    return np.flip(orth_depth, 2).transpose(0, 2, 1)


def clean_up_visible_orth_depth(multilayer_depth, th=16):
    counts = (multilayer_depth != 0).sum((1, 2))
    sparse_layer_ids = counts < th
    result = multilayer_depth[sparse_layer_ids == False]
    if result.shape[0] > 1:
        result = result[:2]
    return result


def get_camera_k_from_angle(img_res, angle):
    mx = my = img_res // 2
    f = img_res / (2 * math.tan(angle / (np.degrees(1) * 2.0)))
    camera_k = np.array([[f, 0, mx], [0, f, my], [0, 0, 1]])
    return camera_k


def adjust_camera_k_by_res(camera_k, original_res, new_res):
    result = camera_k.copy()
    res_change_factor = new_res / original_res

    result[0, 0] *= res_change_factor
    result[1, 1] *= res_change_factor
    result[0, 2] *= res_change_factor
    result[1, 2] *= res_change_factor

    return result



def get_visible_pcd(depth, camera_k, return_type="torch"):
    if isinstance(depth, torch.Tensor):
        device = depth.device
        depth = get_np(depth.squeeze())
    depth_img = depth.squeeze()
    depth_img = np.expand_dims(depth_img, -1)
    depth_img[depth_img != depth_img] = 0

    depth_img_adj = depth_img.copy()

    camera_k = np.array(camera_k)
    visible_pcd = depth_image_to_pointcloud3d(
        depth_image=depth_img_adj, camera_k=camera_k
    )
    points = np.array(visible_pcd).transpose(1, 0)

    if return_type == "torch":
        result = torch.tensor(points, device=device)
    elif return_type == "pv":
        result = pv.PolyData(points)
    elif return_type == "np":
        result = points
    return result



def get_np(t):
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    elif isinstance(t, pv.PolyData):
        return t.points
    else:
        try:
            return np.array(t)
        except:
            return t.get()


def permute(npt, order):
    if isinstance(npt, np.ndarray):
        return np.transpose(npt, order)
    elif isinstance(npt, torch.Tensor):
        return npt.permute(order)


def crop(t, cx, cy):
    # rgb_image
    rgb = False
    if t.shape[-1] == 3:
        rgb = True
        t = permute(t, list(range(0, len(t.shape) - 3)) + [-1, -3, -2])

    if cx == 0 and cy == 0:
        result = t
    elif cx != 0 and cy != 0:
        result = t[..., cx:-cx, cy:-cy]
    elif cx == 0 and cy != 0:
        result = t[..., cy:-cy]
    elif cx != 0 and cy == 0:
        result = t[..., cx:-cx, :]
    if rgb:
        result = permute(result, list(range(0, len(t.shape) - 3)) + [-2, -1, -3])
    return result



def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


def get_indexed_ortho_shell_points(ortho_shell, px_size):
    entry_points, entry_ids = get_ortho_layer_indexed_points(ortho_shell[0], px_size)
    exit_points, exit_ids = get_ortho_layer_indexed_points(ortho_shell[1], px_size)

    size = ortho_shell.shape[-1]
    exit_ids += size ** 2
    points = np.concatenate((entry_points, exit_points))
    point_ids = np.concatenate((entry_ids, exit_ids))
    return points, point_ids


def get_ortho_layer_indexed_points(ortho_layer, px_size):
    size = ortho_layer.shape[-1]
    view_size = size * px_size

    u, v = np.nonzero(ortho_layer)
    z = ortho_layer[u, v]

    x = v * px_size - view_size / 2
    y = u * px_size - view_size / 2

    points = np.stack([x, y, z], axis=-1)
    point_ids = v + u * size

    return points, point_ids


def crop_to_nonzero_data(img_stack, pad=1):
    _, x_data_coords, y_data_coords = img_stack.nonzero()
    x_start = x_data_coords.min()
    y_start = y_data_coords.min()
    x_end = x_data_coords.max()
    y_end = y_data_coords.max()
    x_offset = max(0, x_start - pad)
    y_offset = max(0, y_start - pad)
    cropped_size = max(x_end - x_offset, y_end - y_offset) + pad * 2
    size = img_stack.shape[-1]

    if x_offset + cropped_size > size:
        cropped_size = size - x_offset
    if y_offset + cropped_size > size:
        cropped_size = size - y_offset

    return (
        img_stack[
            :, x_offset : x_offset + cropped_size, y_offset : y_offset + cropped_size
        ],
        x_offset,
        y_offset,
    )


def get_ortho_shell_mesh(
    ortho_shell, px_size, speedcrop=True
) -> o3d.geometry.TriangleMesh:
    original_size = ortho_shell.shape[-1]
    if speedcrop:
        ortho_shell, x_crop, y_crop = crop_to_nonzero_data(ortho_shell)
    else:
        x_crop = 0
        y_crop = 0

    ortho_shell = np.flip(ortho_shell, 1)
    p, p_id = get_indexed_ortho_shell_points(ortho_shell, px_size)
    size = ortho_shell.shape[-1]
    idx_array = np.linspace(0, size ** 2 - 1, size ** 2).reshape([size, size])
    window_shape = (2, 2)
    data_win = [
        rolling_window(ortho_shell[0] != 0, window_shape).reshape(-1, *window_shape),
        rolling_window(ortho_shell[1] != 0, window_shape).reshape(-1, *window_shape),
    ]
    idx_win = (
        rolling_window(idx_array, window_shape)
        .reshape(-1, *window_shape)
        .astype(np.int32)
    )

    face_list = []

    start_idx = 0
    for layer_id in [0, 1]:
        for i in range(idx_win.shape[0]):
            win = data_win[layer_id][i]
            idx = idx_win[i] + start_idx

            if (win != 0).sum() == 4:
                # add both triangles to form a rectangle.
                face_list.append([3] + [idx[0, 0], idx[0, 1], idx[1, 0]])
                face_list.append([3] + [idx[1, 1], idx[0, 1], idx[1, 0]])
            elif (win != 0).sum() == 3:
                # add the present triangle
                face_list.append([3] + idx[win != 0].tolist())
                # add both edge triangles
                if win[0, 0] == 0 or win[1, 1] == 0:
                    edge_idx = [idx[0, 1], idx[1, 0]]
                elif win[0, 1] == 0 or win[1, 0] == 0:
                    edge_idx = [idx[0, 0], idx[1, 1]]
                else:
                    raise Exception()

                if layer_id == 0:
                    face_list.append(
                        [3] + [edge_idx[0], edge_idx[1], edge_idx[1] + size ** 2]
                    )
                    face_list.append(
                        [3]
                        + [
                            edge_idx[0] + size ** 2,
                            edge_idx[1] + size ** 2,
                            edge_idx[0],
                        ]
                    )
            elif (win != 0).sum() == 2:
                if layer_id == 0:
                    edge_idx = idx[win != 0].tolist()
                    face_list.append(
                        [3] + [edge_idx[0], edge_idx[1], edge_idx[1] + size ** 2]
                    )
                    face_list.append(
                        [3]
                        + [
                            edge_idx[0] + size ** 2,
                            edge_idx[1] + size ** 2,
                            edge_idx[0],
                        ]
                    )

        start_idx += size ** 2

    faces = np.array(face_list)
    remapped_faces = npi.remap(
        faces[:, 1:].flatten(), p_id.tolist(), list(range(p_id.shape[0]))
    )
    remapped_faces = remapped_faces.reshape(faces[:, 1:].shape)
    faces[:, 1:] = remapped_faces

    origin_offset = (original_size - size) / 2
    p[:, 0] += (y_crop - origin_offset) * px_size
    p[:, 1] -= (x_crop - origin_offset) * px_size

    vertices = o3d.utility.Vector3dVector(p)
    faces = o3d.utility.Vector3iVector(faces[:, 1:])
    return o3d.geometry.TriangleMesh(vertices, faces)


def persp2ortho_shell(persp_shell, px_size, camera_k):
    ortho_1 = persp2ortho(persp_shell[0], px_size, camera_k, upsample_factor=2)
    exit_shell = persp_shell[1].copy()
    exit_shell[exit_shell > 0] = 0
    exit_shell[exit_shell < -2] = -2
    ortho_2 = persp2ortho(exit_shell, px_size, camera_k, upsample_factor=2)
    stacked = np.concatenate([ortho_1, ortho_2])

    result_exit = stacked.min(0)
    zero_mask = stacked == 0
    stacked[zero_mask] = float("-inf")
    result_entry = stacked.max(0)
    result_entry[result_entry == float("-inf")] = 0
    return np.stack([result_entry, result_exit])


def get_persp_shell_mesh(persp_shell, px_size, camera_k):
    ortho_shell = persp2ortho_shell(persp_shell, px_size, camera_k)
    result = get_ortho_shell_mesh(ortho_shell, px_size)
    return result


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def camera_k_and_shape_to_intrinsic(shape, camera_k):
    return o3d.camera.PinholeCameraIntrinsic(
        shape[0],
        shape[1],
        camera_k[0, 0],
        camera_k[1, 1],
        camera_k[0, 2],
        camera_k[1, 2],
    )


def cropped_o3dpc(
    o3dpcd,
    bbox_minlims: List[float] = None,  # len 3
    bbox_maxlims: List[float] = None,  # len 3
):
    if (
        bbox_minlims is not None
        and len(bbox_minlims) == 3
        and bbox_maxlims is not None
        and len(bbox_maxlims) == 3
    ):
        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox_minlims, max_bound=bbox_maxlims
        )
        return o3dpcd.crop(crop_bbox)
    return o3dpcd


def o3d_pointcloud_native(
    depth_img: np.ndarray,
    camera_k: np.ndarray,
    extrinsic: np.ndarray = None,
    scale_factor: float = 1000.0,
    depth_trunc: float = 1000.0,
    bbox_minlims: List[float] = None,
    bbox_maxlims: List[float] = None,
):
    """
    Given a depth image and camera_k, creates
    an open3d point cloud. The depth image
    may be already masked, in which case depth
    values of 0 mean they have been masked out
    (or are actually 0).
    """

    depth_o3dimg = o3d.geometry.Image(depth_img)

    # build open3d camera
    cam_intrinsic = camera_k_and_shape_to_intrinsic(depth_img.shape, camera_k)

    # generate open3d pointcloud from depth image
    o3dpcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3dimg,
        cam_intrinsic,
        depth_scale=scale_factor,
        depth_trunc=depth_trunc,
        project_valid_depth_only=True,
    )

    if extrinsic is not None:
        o3dpcd = o3dpcd.transform(extrinsic)

    _, in_ind = o3dpcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    o3dpcd = o3dpcd.select_by_index(in_ind)
    return cropped_o3dpc(o3dpcd, bbox_minlims, bbox_maxlims)


def camera_points_to_depth_coords(
    pc: np.ndarray, camera_k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a pointcloud in the camera frame to a set of coordinates and depth values in the image frame.

    Args:
        pc: The PointCloud to transform.
        camera_k: The camera intrinsics matrix.

    Returns:
        A tuple of (coordinates [u,v], depth values).
    """

    # Normalize the array by the Z coordinates.
    z_values = pc[2, :]
    norm_arr = np.divide(pc, z_values)

    # Multiply the intrinsics matrix by the normalized points.
    coords = camera_k @ norm_arr

    return coords[0:2], z_values


def pointcloud3d_to_depth_image(
    pc: np.ndarray, img_width: int, img_height: int, camera_k: np.ndarray
) -> np.ndarray:
    """Convert a PointCloud3D to a depth image (HWC)

    Args:
        camera_k: Camera intrinsics. Must be the same units as the depth image, so likely mm
        img_width: width in pixels (note this must be sane values for the camera_k)
        img_height: height in pixels (note this must be a sane value for the camera_k)


    Returns:
        depth image as a numpy array (HWC) in the units of the pointcloud and camera_k
    """
    # ignore points at 0
    d_zero_mask = pc[2, :] != 0
    pc = pc[:, d_zero_mask]

    coords, zs = camera_points_to_depth_coords(pc, camera_k)
    new_di = np.zeros((img_height, img_width, 1))  # depth image is a 1 channel image

    # u => j
    # v => i
    coords_i = np.floor(coords[1]).astype(int)
    coords_j = np.floor(coords[0]).astype(int)

    # Bound the coordinates to valid values in the image.
    # i_in_bounds = np.logical_and(0 <= coords_i, coords_i < height)
    # j_in_bounds = np.logical_and(0 <= coords_j, coords_j < width)
    i_in_bounds = np.logical_and(0 <= coords_i, coords_i < new_di.shape[0])
    j_in_bounds = np.logical_and(0 <= coords_j, coords_j < new_di.shape[1])
    clipped_ixs = np.where(np.logical_and(i_in_bounds, j_in_bounds))[0]
    clipped_is = coords_i[clipped_ixs]
    clipped_js = coords_j[clipped_ixs]

    # Create a new, resampled depth image
    new_di[clipped_is, clipped_js] = zs[clipped_ixs].reshape((-1, 1))

    # Eliminate resampling artifacts for re-projected depth image
    filtered_new_di = scipy.ndimage.maximum_filter(new_di.squeeze(), size=2)
    new_di[new_di == 0] = filtered_new_di[(new_di == 0).squeeze()]

    new_di = new_di[:, :, 0]
    return new_di


def get_masked_depth(
    object_id: int, mask: np.ndarray, depth: np.ndarray,
) -> np.ndarray:
    """
    Given a mask, the object_id of an object in the mask,
    and a depth image, returns the depth image
    such that any points not in the mask are set to 0.

    Args:
        object_id: Id of the object in the mask we are masking out.
        mask: The mask - must be same size as depth
        depth: the depth image

    Returns:
        The depth image masked for the given mask and object id
    """

    return depth * (mask == object_id).astype(np.float)
