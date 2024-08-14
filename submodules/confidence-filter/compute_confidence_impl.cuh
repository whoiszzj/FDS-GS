#ifndef COMPUTE_CONFIDENCE_IMPL_CUH
#define COMPUTE_CONFIDENCE_IMPL_CUH

#include "auxiliary.h"

void compute_confidence_sample_cuda(
        const int N,
        const int H1,
        const int W1,
        const int H2,
        const int W2,
        const float *xyz,
        const float *cov,
        const float *K_1,
        const float *K_2,
        const float *Rt_1,
        const float *Rt_2,
        const float *img_1,
        const float *img_2,
        float *confidence
);

#endif //COMPUTE_CONFIDENCE_IMPL_CUH