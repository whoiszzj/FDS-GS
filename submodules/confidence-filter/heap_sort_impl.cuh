#ifndef GEOM_GUIDE_HEAP_SORT_IMPL_CUH
#define GEOM_GUIDE_HEAP_SORT_IMPL_CUH

#include "auxiliary.h"

typedef struct {
    float contribution;
    int index;
} Cont_index;

void heap_sort_cuda(
        const int N,
        float *contribution_heap,
        int *index_heap,
        const float *contribution,
        const int index,
        const int topk
);

#endif //GEOM_GUIDE_HEAP_SORT_IMPL_CUH
