#include "heap_sort_impl.cuh"

__device__ void swap(float *confidence_heap_a, float *confidence_heap_b, int *index_heap_a, int *index_heap_b) {
    float tmp_confidence = *confidence_heap_a;
    *confidence_heap_a = *confidence_heap_b;
    *confidence_heap_b = tmp_confidence;
    int tmp_index = *index_heap_a;
    *index_heap_a = *index_heap_b;
    *index_heap_b = tmp_index;
}

__device__ void heapify_down(float *confidence_heap, int *index_heap, int size, int index) {
    while (1) {
        int smallest = index;
        int left = 2 * index + 1;
        int right = 2 * index + 2;

        if (left < size && confidence_heap[left] < confidence_heap[smallest]) {
            smallest = left;
        }
        if (right < size && confidence_heap[right] < confidence_heap[smallest]) {
            smallest = right;
        }
        if (smallest != index) {
            swap(&confidence_heap[index], &confidence_heap[smallest], &index_heap[index], &index_heap[smallest]);
            index = smallest;
        } else {
            break;
        }
    }
}

__device__ void heapify_up(float *confidence_heap, int *index_heap, int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (confidence_heap[parent]> confidence_heap[index]) {
            swap(&confidence_heap[parent], &confidence_heap[index], &index_heap[parent], &index_heap[index]);
            index = parent;
        } else {
            break;
        }
    }
}

__device__ void
insert_heap(float *confidence_heap, int *index_heap, const int topk, int *size, float index, float contribution) {
    if (*size < topk) {
        index_heap[*size] = index;
        confidence_heap[*size] = contribution;
        heapify_up(confidence_heap, index_heap, *size);
        (*size)++;
    } else if (contribution > confidence_heap[0]) {
        index_heap[0] = index;
        confidence_heap[0] = contribution;
        heapify_down(confidence_heap, index_heap, *size, 0);
    }
}

__global__ void heap_sort_kernel(
        const int N,
        float *contribution_heap,
        int *index_heap,
        const float *contribution,
        const int index,
        const int topk
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    int size = 0;
    for (int i = 0; i < topk; ++i) {
        if (index_heap[tid * topk + i] != -1) {
            size++;
        }
    }
    insert_heap(contribution_heap + tid * topk, index_heap + tid * topk, topk, &size, index, contribution[tid]);
}


void heap_sort_cuda(
        const int N,
        float *contribution_heap,
        int *index_heap,
        const float *contribution,
        const int index,
        const int topk
) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    heap_sort_kernel << < grid_size, block_size >> > (N, contribution_heap, index_heap, contribution, index, topk);
    CHECK_CUDA;
}
