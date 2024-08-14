import cv2
import numpy as np
import os

import torch
from matplotlib import pyplot as plt
from plyfile import PlyElement, PlyData
from confidence_filter import compute_confidence_sample


def project_Gaussians_to_image(xyz, intrinsics, extrinsics):
    # xyz [N, 3]: torch tensor
    # intrinsics [3, 3]: torch tensor => R | t
    # extrinsics [4, 4]: torch tensor => K
    # return: [N, 2]
    xyz_cam = torch.mm(extrinsics[0:3, 0:3], xyz.t()) + extrinsics[0:3, 3].unsqueeze(1)  # [3, N]
    uv = torch.mm(intrinsics, xyz_cam)  # [3, N]
    uv = uv / (uv[2, :] + 1e-8)  # [3, N]
    return uv[0:2, :].t()  # [N, 2]


def projection_in_image(xyz, intrinsics, extrinsics, H, W):
    # xyz [N, 3]: torch tensor
    # intrinsics [3, 3]: torch tensor => R | t
    # extrinsics [4, 4]: torch tensor => K
    # H: int
    # W: int
    # return: [N]
    uv = project_Gaussians_to_image(xyz, intrinsics, extrinsics)  # [N, 2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return mask


def compute_confidence_sample_wrapper(xyz, cov, cam1, cam2):
    # xyz [N, 3]: torch tensor
    # cam1: Camera
    # cam2: Camera
    # return: [N]
    cam1_ncc_dict = cam1.ncc_dict
    cam2_ncc_dict = cam2.ncc_dict
    scale = 1.0

    K_1 = cam1_ncc_dict[scale]["intrinsics"]
    K_2 = cam2_ncc_dict[scale]["intrinsics"]
    Rt_1 = cam1_ncc_dict[scale]["extrinsics"]
    Rt_2 = cam2_ncc_dict[scale]["extrinsics"]
    img_1 = cam1_ncc_dict[scale]["gray_image"]
    img_2 = cam2_ncc_dict[scale]["gray_image"]
    confidence = compute_confidence_sample(xyz, cov, K_1, K_2, Rt_1, Rt_2, img_1, img_2)
    return confidence
