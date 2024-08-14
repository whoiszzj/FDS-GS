#ifndef CONFIDENCE_FILTER_H
#define CONFIDENCE_FILTER_H

#include <torch/extension.h>
#include "compute_confidence_impl.cuh"
#include "heap_sort_impl.cuh"

void compute_confidence_sample_wrapper(
        const torch::Tensor &xyz,  // (N, 3)
        const torch::Tensor &cov,  // (N, 3)
        const torch::Tensor &K_1,
        const torch::Tensor &K_2,
        const torch::Tensor &Rt_1,
        const torch::Tensor &Rt_2,
        const torch::Tensor &img_1,
        const torch::Tensor &img_2,
        torch::Tensor &confidence
);

void heap_sort_wrapper(
        torch::Tensor &contribution_heap,
        torch::Tensor &index_heap,
        const torch::Tensor &contribution,
        const int index,
        const int topk
);

#endif //CONFIDENCE_FILTER_H