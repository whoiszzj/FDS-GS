#include "compute_confidence_impl.cuh"
#include <glm/glm.hpp>


__device__ float2 project_3d_points(
        const float *K,
        const float *Rt,
        const float3 xyz
) {
    float3 xyz_rot; // R @ xyz + t
    xyz_rot.x = Rt[0] * xyz.x + Rt[1] * xyz.y + Rt[2] * xyz.z + Rt[3];
    xyz_rot.y = Rt[4] * xyz.x + Rt[5] * xyz.y + Rt[6] * xyz.z + Rt[7];
    xyz_rot.z = Rt[8] * xyz.x + Rt[9] * xyz.y + Rt[10] * xyz.z + Rt[11];

    float depth = K[6] * xyz_rot.x + K[7] * xyz_rot.y + K[8] * xyz_rot.z;

    if (depth <= 0.0f) {
        return;
    }
    float2 uv;
    uv.x = K[0] * xyz_rot.x + K[1] * xyz_rot.y + K[2] * xyz_rot.z;
    uv.y = K[3] * xyz_rot.x + K[4] * xyz_rot.y + K[5] * xyz_rot.z;
    uv.x /= depth;
    uv.y /= depth;
    return uv;
}

__device__ glm::mat2 compute_conv_2D(
        const int N,
        const int height,
        const int width,
        const float *K,
        const float *Rt,
        const float3 pos_world,
        const float *conv3d
) {
    float3 t = make_float3(
            Rt[0] * pos_world.x + Rt[1] * pos_world.y + Rt[2] * pos_world.z + Rt[3],
            Rt[4] * pos_world.x + Rt[5] * pos_world.y + Rt[6] * pos_world.z + Rt[7],
            Rt[8] * pos_world.x + Rt[9] * pos_world.y + Rt[10] * pos_world.z + Rt[11]
    );
    const float focal_x = K[0];
    const float focal_y = K[4];
//    float fovx = 2 * atan(W / (2 * focal_x));
//    float fovy = 2 * atan(H / (2 * focal_y));
//    float tan_fovx = tan(fovx / 2);
//    float tan_fovy = tan(fovy / 2);
    const float tan_fovx = width / (2 * focal_x);
    const float tan_fovy = height / (2 * focal_y);
    const float limx = 1.3 * tan_fovx;
    const float limy = 1.3 * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
            focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
            0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0, 0, 0
    );

    glm::mat3 W = glm::mat3(
            Rt[0], Rt[1], Rt[2],
            Rt[4], Rt[5], Rt[6],
            Rt[8], Rt[9], Rt[10]
    );

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
            conv3d[0], conv3d[1], conv3d[2],
            conv3d[1], conv3d[3], conv3d[4],
            conv3d[2], conv3d[4], conv3d[5]
    );

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    glm::mat2 conv2d = glm::mat2(
            cov[0][0], cov[0][1],
            cov[1][0], cov[1][1]
    );
    glm::mat2 L = glm::mat2(0, 0, 0, 0);
    L[0][0] = sqrt(conv2d[0][0]);
    L[0][1] = 0;
    L[1][0] = conv2d[1][0] / L[0][0];
    L[1][1] = sqrt(conv2d[1][1] - L[1][0] * L[1][0]);
    return L;
}

__global__ void compute_confidence_sample_kernel(
        const int N,
        const int H1,
        const int W1,
        const int H2,
        const int W2,
        const float *xyz,
        const float *cov,
        const float *sample_points,
        const float *K_1,
        const float *K_2,
        const float *Rt_1,
        const float *Rt_2,
        const float *img_1,
        const float *img_2,
        float *confidence
) {
    // point idx
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const float3 pos_world = make_float3(xyz[idx * 3 + 0], xyz[idx * 3 + 1], xyz[idx * 3 + 2]);
    const float2 uv_1 = project_3d_points(K_1, Rt_1, pos_world);
    const float2 uv_2 = project_3d_points(K_2, Rt_2, pos_world);

    if (uv_1.x < 0 || uv_1.x >= W1 || uv_1.y < 0 || uv_1.y >= H1 ||
        uv_2.x < 0 || uv_2.x >= W2 || uv_2.y < 0 || uv_2.y >= H2) {
        confidence[idx] = 0.0f;
        return;
    }
    const float *cur_cov = cov + idx * 6;

    const glm::mat2 conv_1_L = compute_conv_2D(N, H1, W1, K_1, Rt_1, pos_world, cur_cov);
    const glm::mat2 conv_2_L = compute_conv_2D(N, H2, W2, K_2, Rt_2, pos_world, cur_cov);

    const int sample_num = 49;
    float mu1 = 0.0f;
    float mu2 = 0.0f;
    float sigma1_sq = 0.0f;
    float sigma2_sq = 0.0f;
    float sigma12 = 0.0f;
    float weight = 0.0f;

    const glm::vec2 uv_1_glm = glm::vec2(uv_1.x, uv_1.y);
    const glm::vec2 uv_2_glm = glm::vec2(uv_2.x, uv_2.y);

    for (int i = 0; i < sample_num; ++i) {
        const glm::vec2 std_sample_point = glm::vec2(sample_points[i * 3 + 0], sample_points[i * 3 + 1]);
        const float sample_weight = sample_points[i * 3 + 2];
        const glm::vec2 ref_sample_point_glm = conv_1_L * std_sample_point + uv_1_glm;
        const glm::vec2 src_sample_point_glm = conv_2_L * std_sample_point + uv_2_glm;
        const int2 ref_sample_point = make_int2(int(ref_sample_point_glm.x), int(ref_sample_point_glm.y));
        const int2 src_sample_point = make_int2(int(src_sample_point_glm.x), int(src_sample_point_glm.y));

        if (ref_sample_point.x >= W1 || ref_sample_point.x < 0 ||
            ref_sample_point.y >= H1 || ref_sample_point.y < 0) {
            continue;
        }
        if (src_sample_point.x >= W2 || src_sample_point.x < 0 ||
            src_sample_point.y >= H2 || src_sample_point.y < 0) {
            continue;
        }
        const int ref_sample_idx = ref_sample_point.y * W1 + ref_sample_point.x;
        const int src_sample_idx = src_sample_point.y * W2 + src_sample_point.x;

        const float ref_sample_color = img_1[ref_sample_idx];
        const float src_sample_color = img_2[src_sample_idx];
        weight += sample_weight;
        mu1 += ref_sample_color * sample_weight;
        mu2 += src_sample_color * sample_weight;
        sigma1_sq += ref_sample_color * ref_sample_color * sample_weight;
        sigma2_sq += src_sample_color * src_sample_color * sample_weight;
        sigma12 += ref_sample_color * src_sample_color * sample_weight;

    }
    if (weight <= 0) {
        confidence[idx] = 0.0f;
        return;
    }
    mu1 /= weight;
    mu2 /= weight;
    float mu1_sq = mu1 * mu1;
    float mu2_sq = mu2 * mu2;
    float mu1_mu2 = mu1 * mu2;
    sigma1_sq = sigma1_sq / weight - mu1_sq;
    sigma2_sq = sigma2_sq / weight - mu2_sq;
    sigma12 = sigma12 / weight - mu1_mu2;

    const float C1 = 0.01 * 255 * 0.01 * 255;
    const float C2 = 0.03 * 255 * 0.03 * 255;
    float ssim = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

//    printf("mu1: %f, mu2: %f, sigma1_sq: %f, sigma2_sq: %f, sigma12: %f, ssim: %f\n", mu1, mu2, sigma1_sq, sigma2_sq, sigma12, ssim);

    if (ssim < 0.0f) {
        ssim = 0.0f;
    }
    if (ssim > 1.0f) {
        ssim = 1.0f;
    }
    confidence[idx] = ssim;
}


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
) {
    float *sample_points_cuda;
    cudaMalloc(&sample_points_cuda, 49 * 3 * sizeof(float));
    cudaMemcpy(sample_points_cuda, sample_points, 49 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    compute_confidence_sample_kernel <<< (N + 255) / 256, 256 >>>(
            N, H1, W1, H2, W2, xyz, cov, sample_points_cuda, K_1, K_2, Rt_1, Rt_2, img_1, img_2, confidence
    );
    CHECK_CUDA;
    cudaFree(sample_points_cuda);
}