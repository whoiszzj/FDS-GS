import torch
from . import _C


def compute_confidence_sample(xyz, cov, K_1, K_2, Rt_1, Rt_2, img_1, img_2):
    # xyz: torch.Tensor with shape [N, 3]
    # K_1: torch.Tensor with shape [3, 3]
    # K_2: torch.Tensor with shape [3, 3]
    # Rt_1: torch.Tensor with shape [4, 4]
    # Rt_2: torch.Tensor with shape [4, 4]
    # img_1: torch.Tensor with shape [H, W]
    # img_2: torch.Tensor with shape [H, W]
    gaussian_num = xyz.shape[0]
    confidence = torch.zeros((gaussian_num,), dtype=torch.float32).cuda()
    _C.compute_confidence_sample(xyz, cov, K_1, K_2, Rt_1, Rt_2, img_1, img_2, confidence)
    return confidence


def heap_sort(contribution_heap, index_heap, contribution, index, topk=2):
    # contribution_heap: torch.Tensor with shape [N, topk]
    # index_heap: torch.Tensor with shape [N, topk]
    # contribution: torch.Tensor with shape [N]
    # index: int
    # topk: int
    _C.heap_sort(contribution_heap, index_heap, contribution, index, topk)
    return contribution_heap, index_heap
