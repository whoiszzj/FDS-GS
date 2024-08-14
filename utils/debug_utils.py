import os
import numpy as np
from plyfile import PlyData, PlyElement


def save_deleted_gaussians_confidence(
        path,
        original_xyz,
        confidence_mask,
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    original_xyz = original_xyz.detach().cpu().numpy()
    confidence_mask = confidence_mask.detach().cpu().numpy()
    xyzrgb = np.zeros((original_xyz.shape[0],), dtype=dtype)
    xyzrgb['x'] = original_xyz[:, 0]
    xyzrgb['y'] = original_xyz[:, 1]
    xyzrgb['z'] = original_xyz[:, 2]
    xyzrgb['red'] = 255
    xyzrgb['green'] = 255
    xyzrgb['blue'] = 255
    xyzrgb['green'][confidence_mask] = 0  # the points that are filtered by confident are (green mix blue) = cyan
    # the points that are filtered by both are (blue) = blue
    el = PlyElement.describe(xyzrgb, 'vertex')
    PlyData([el]).write(os.path.join(path, "deleted_gaussians_confidence.ply"))


def save_deleted_gaussians(
        path,
        original_xyz,
        mask,
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    original_xyz = original_xyz.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    xyzrgb = np.zeros((original_xyz.shape[0],), dtype=dtype)
    xyzrgb['x'] = original_xyz[:, 0]
    xyzrgb['y'] = original_xyz[:, 1]
    xyzrgb['z'] = original_xyz[:, 2]
    xyzrgb['red'] = 255
    xyzrgb['green'] = 255
    xyzrgb['blue'] = 255
    xyzrgb['green'][mask] = 0  # the points that are filtered by confident are (green mix blue) = cyan
    # the points that are filtered by both are (blue) = blue
    el = PlyElement.describe(xyzrgb, 'vertex')
    PlyData([el]).write(os.path.join(path, "deleted_gaussians.ply"))


def save_added_gaussians(
        path,
        xyz,
        add_mask
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    xyz = xyz.detach().cpu().numpy()
    add_mask = add_mask.detach().cpu().numpy()
    xyzrgb = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzrgb['x'] = xyz[:, 0]
    xyzrgb['y'] = xyz[:, 1]
    xyzrgb['z'] = xyz[:, 2]
    xyzrgb['red'] = 255
    xyzrgb['green'] = 255
    xyzrgb['blue'] = 255
    xyzrgb['red'][add_mask] = 0
    xyzrgb['blue'][add_mask] = 0

    # the points that are filtered by both are (blue) = blue
    el = PlyElement.describe(xyzrgb, 'vertex')
    PlyData([el]).write(os.path.join(path, "added_gaussians.ply"))


def save_confidence(
        path,
        xyz,
        confidence,
        vis_pair
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('confidence', 'f4'), ('vis_1', 'i4'), ('vis_2', 'i4')
    ]
    xyz = xyz.detach().cpu().numpy()
    xyzc = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzc['x'] = xyz[:, 0]
    xyzc['y'] = xyz[:, 1]
    xyzc['z'] = xyz[:, 2]
    xyzc['confidence'] = confidence.detach().cpu().numpy()
    vis_pair = vis_pair.detach().cpu().numpy()
    xyzc['vis_1'] = vis_pair[:, 0]
    xyzc['vis_2'] = vis_pair[:, 1]
    el = PlyElement.describe(xyzc, 'vertex')
    PlyData([el]).write(os.path.join(path, "confidence.ply"))


def save_opacity(
        path,
        xyz,
        opacity
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4')
    ]
    xyz = xyz.detach().cpu().numpy()
    xyzc = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzc['x'] = xyz[:, 0]
    xyzc['y'] = xyz[:, 1]
    xyzc['z'] = xyz[:, 2]
    xyzc['opacity'] = opacity.squeeze(-1).detach().cpu().numpy()
    el = PlyElement.describe(xyzc, 'vertex')
    PlyData([el]).write(os.path.join(path, "opacity.ply"))


def save_viewspace_gradient(
        path,
        xyz,
        gradient
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('gradient', 'f4')
    ]
    xyz = xyz.detach().cpu().numpy()
    xyzg = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzg['x'] = xyz[:, 0]
    xyzg['y'] = xyz[:, 1]
    xyzg['z'] = xyz[:, 2]
    xyzg['gradient'] = gradient.detach().cpu().numpy()
    el = PlyElement.describe(xyzg, 'vertex')
    PlyData([el]).write(os.path.join(path, "viewspace_gradient.ply"))


def save_contribution_weight(
        path,
        xyz,
        contribution_weight
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('contribution_weight', 'f4')
    ]
    xyz = xyz.detach().cpu().numpy()
    xyzcw = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzcw['x'] = xyz[:, 0]
    xyzcw['y'] = xyz[:, 1]
    xyzcw['z'] = xyz[:, 2]
    xyzcw['contribution_weight'] = contribution_weight.detach().cpu().numpy()
    el = PlyElement.describe(xyzcw, 'vertex')
    PlyData([el]).write(os.path.join(path, "contribution_weight.ply"))


def save_densified_points(
        path,
        old_xyz,  # [N, 3]
        new_xyz  # [N, 2, 3]
):
    os.makedirs(path, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    old_xyz = old_xyz.detach().cpu().numpy()
    new_xyz = new_xyz.detach().cpu().numpy()
    N = old_xyz.shape[0]
    color = np.random.randint(0, 255, (N, 3), dtype=np.uint8)  # [N, 3]
    xyz = np.concatenate([old_xyz[:, None, :], new_xyz], axis=1)  # [N, 3, 3]
    # repeat the color for each point
    color = np.repeat(color[:, None, :], 3, axis=1)  # [N, 3, 3]
    xyz = xyz.reshape(-1, 3)  # [3N, 3]
    color = color.reshape(-1, 3)  # [3N, 3]
    xyzrgb = np.zeros((xyz.shape[0],), dtype=dtype)
    xyzrgb['x'] = xyz[:, 0]
    xyzrgb['y'] = xyz[:, 1]
    xyzrgb['z'] = xyz[:, 2]
    xyzrgb['red'] = color[:, 0]
    xyzrgb['green'] = color[:, 1]
    xyzrgb['blue'] = color[:, 2]
    el = PlyElement.describe(xyzrgb, 'vertex')
    PlyData([el]).write(os.path.join(path, "densified_points.ply"))
