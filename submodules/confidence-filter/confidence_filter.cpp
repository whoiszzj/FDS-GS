#include "confidence_filter.h"

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
) {
    // define middle variables
    const int N = xyz.size(0);
    const int H1 = img_1.size(0);
    const int W1 = img_1.size(1);
    const int H2 = img_2.size(0);
    const int W2 = img_2.size(1);
    compute_confidence_sample_cuda(
            N,
            H1,
            W1,
            H2,
            W2,
            xyz.data_ptr<float>(), //
            cov.data_ptr<float>(), // Cuda
            K_1.data_ptr<float>(), // Cuda
            K_2.data_ptr<float>(), // Cuda
            Rt_1.data_ptr<float>(), // Cuda
            Rt_2.data_ptr<float>(), // Cuda
            img_1.data_ptr<float>(), // Cuda
            img_2.data_ptr<float>(), // Cuda
            confidence.data_ptr<float>()
    ); // Cuda
}

void heap_sort_wrapper(
        torch::Tensor &contribution_heap,
        torch::Tensor &index_heap,
        const torch::Tensor &contribution,
        const int index,
        const int topk
) {
    const int N = contribution_heap.size(0);
    heap_sort_cuda(
            N,
            contribution_heap.data_ptr<float>(), // Cuda
            index_heap.data_ptr<int>(), // Cuda
            contribution.data_ptr<float>(), // Cuda
            index,
            topk
    ); // Cuda
}