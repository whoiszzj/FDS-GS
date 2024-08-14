#ifndef AUXILIARY_H
#define AUXILIARY_H
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
// Includes STD libs
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
#include <cassert>
#include <math.h>
#include <cstdio>
#include <sstream>
#include <tuple>
#include <stdio.h>
#include <functional>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define NEIGHBOR_NUM 8
// DEBUG
#define DEBUG true
#define DEBUG_POINT_IDX 0

#define CHECK_CUDA \
{\
    auto ret = cudaDeviceSynchronize(); \
    if (ret != cudaSuccess) { \
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "(line " << __LINE__ << ") in Function: " << __FUNCTION__ << "\n" << cudaGetErrorString(ret); \
        throw std::runtime_error(cudaGetErrorString(ret)); \
    }\
}
#define M_PI 3.14159265358979323846

const float sample_points[] = {
        0.0f, 0.0f, 1.0f,
        0.000f, 0.500f, 0.882f,
        0.500f, 0.000f, 0.882f,
        0.000f, -0.500f, 0.882f,
        -0.500f, 0.000f, 0.882f,
        0.354f, 0.354f, 0.882f,
        0.354f, -0.354f, 0.882f,
        -0.354f, 0.354f, 0.882f,
        -0.354f, -0.354f, 0.882f,
        0.000f, 1.000f, 0.607f,
        1.000f, 0.000f, 0.607f,
        0.000f, -1.000f, 0.607f,
        -1.000f, 0.000f, 0.607f,
        0.707f, 0.707f, 0.607f,
        0.707f, -0.707f, 0.607f,
        -0.707f, 0.707f, 0.607f,
        -0.707f, -0.707f, 0.607f,
        0.000f, 1.500f, 0.325f,
        1.500f, 0.000f, 0.325f,
        0.000f, -1.500f, 0.325f,
        -1.500f, 0.000f, 0.325f,
        1.061f, 1.061f, 0.325f,
        1.061f, -1.061f, 0.325f,
        -1.061f, 1.061f, 0.325f,
        -1.061f, -1.061f, 0.325f,
        0.000f, 2.000f, 0.135f,
        2.000f, 0.000f, 0.135f,
        0.000f, -2.000f, 0.135f,
        -2.000f, 0.000f, 0.135f,
        1.414f, 1.414f, 0.135f,
        1.414f, -1.414f, 0.135f,
        -1.414f, 1.414f, 0.135f,
        -1.414f, -1.414f, 0.135f,
        0.000f, 2.500f, 0.044f,
        2.500f, 0.000f, 0.044f,
        0.000f, -2.500f, 0.044f,
        -2.500f, 0.000f, 0.044f,
        1.768f, 1.768f, 0.044f,
        1.768f, -1.768f, 0.044f,
        -1.768f, 1.768f, 0.044f,
        -1.768f, -1.768f, 0.044f,
        0.000f, 3.000f, 0.011f,
        3.000f, 0.000f, 0.011f,
        0.000f, -3.000f, 0.011f,
        -3.000f, 0.000f, 0.011f,
        2.121f, 2.121f, 0.011f,
        2.121f, -2.121f, 0.011f,
        -2.121f, 2.121f, 0.011f,
        -2.121f, -2.121f, 0.011f
};

#endif //AUXILIARY_H
