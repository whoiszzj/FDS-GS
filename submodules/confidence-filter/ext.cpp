#include <torch/extension.h>
#include "confidence_filter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("compute_confidence_sample", &compute_confidence_sample_wrapper, "Compute confidence");
m.def("heap_sort", &heap_sort_wrapper, "Heap sort");
}