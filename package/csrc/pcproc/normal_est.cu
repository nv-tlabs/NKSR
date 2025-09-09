#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

#include "cuda_kdtree.cuh"
#include "pcproc.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

__device__ __forceinline__ float4 sym3eig(float3& x1, float3& x2, float3& x3) {
    float4 ret_val;

    const float p1 = x1.y * x1.y + x1.z * x1.z + x2.z * x2.z;
    const float q = (x1.x + x2.y + x3.z) / 3.0f;
    const float p2 = (x1.x - q) * (x1.x - q) + (x2.y - q) * (x2.y - q) + (x3.z - q) * (x3.z - q) + 2 * p1;
    const float p = sqrt(p2 / 6.0f);
    const float b11 = (1.0f / p) * (x1.x - q); const float b12 = (1.0f / p) * x1.y;
    const float b13 = (1.0f / p) * x1.z; const float b21 = (1.0f / p) * x2.x;
    const float b22 = (1.0f / p) * (x2.y - q); const float b23 = (1.0f / p) * x2.z;
    const float b31 = (1.0f / p) * x3.x; const float b32 = (1.0f / p) * x3.y;
    const float b33 = (1.0f / p) * (x3.z - q);

    float r = b11 * b22 * b33 + b12 * b23 * b31 + b13 * b21 * b32 -
              b13 * b22 * b31 - b12 * b21 * b33 - b11 * b23 * b32;
    r = r / 2.0f;

    float phi;
    if (r <= -1) {
        phi = M_PI / 3.0f;
    } else if (r >= 1) {
        phi = 0;
    } else {
        phi = acos(r) / 3.0f;
    }

//    float e0 = q + 2 * p * cos(phi);
    ret_val.w = q + 2 * p * cos(phi + (2 * M_PI / 3));
//    ret_val.w = 3 * q - e0 - e1;

    x1.x -= ret_val.w;
    x2.y -= ret_val.w;
    x3.z -= ret_val.w;

    const float r12_1 = x1.y * x2.z - x1.z * x2.y;
    const float r12_2 = x1.z * x2.x - x1.x * x2.z;
    const float r12_3 = x1.x * x2.y - x1.y * x2.x;
    const float r13_1 = x1.y * x3.z - x1.z * x3.y;
    const float r13_2 = x1.z * x3.x - x1.x * x3.z;
    const float r13_3 = x1.x * x3.y - x1.y * x3.x;
    const float r23_1 = x2.y * x3.z - x2.z * x3.y;
    const float r23_2 = x2.z * x3.x - x2.x * x3.z;
    const float r23_3 = x2.x * x3.y - x2.y * x3.x;

    const float d1 = r12_1 * r12_1 + r12_2 * r12_2 + r12_3 * r12_3;
    const float d2 = r13_1 * r13_1 + r13_2 * r13_2 + r13_3 * r13_3;
    const float d3 = r23_1 * r23_1 + r23_2 * r23_2 + r23_3 * r23_3;

    float d_max = d1;
    int i_max = 0;

    if (d2 > d_max) {
        d_max = d2;
        i_max = 1;
    }

    if (d3 > d_max) {
        i_max = 2;
    }

    if (i_max == 0) {
        ret_val.x = r12_1 / sqrt(d1);
        ret_val.y = r12_2 / sqrt(d1);
        ret_val.z = r12_3 / sqrt(d1);
    } else if (i_max == 1) {
        ret_val.x = r13_1 / sqrt(d2);
        ret_val.y = r13_2 / sqrt(d2);
        ret_val.z = r13_3 / sqrt(d2);
    } else {
        ret_val.x = r23_1 / sqrt(d3);
        ret_val.y = r23_2 / sqrt(d3);
        ret_val.z = r23_3 / sqrt(d3);
    }

    return ret_val;
}

__global__ void estimate_normal_kernel(
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_pc,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_nn_dist,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> input_nn_ind,
        int max_nn, float radius,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_normal) {

    const int pc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pc_id >= input_pc.size(0)) {
        return;
    }

    float3 pc_mean{0.0f, 0.0f, 0.0f};
    float valid_count = 0.0f;
    for (int nn_i = 1; nn_i < max_nn; ++nn_i) {
        if (radius < 0.0 || input_nn_dist[pc_id][nn_i] < radius * radius) {
            int nn_pc_id = input_nn_ind[pc_id][nn_i];
            pc_mean += make_float3(input_pc[nn_pc_id][0], input_pc[nn_pc_id][1], input_pc[nn_pc_id][2]);
            valid_count += 1.0f;
        } else break;
    }
    pc_mean /= valid_count;

    // Compute the covariance matrix.
    float3 cov_x1{0.0f, 0.0f, 0.0f};
    float3 cov_x2{0.0f, 0.0f, 0.0f};
    float3 cov_x3{0.0f, 0.0f, 0.0f};
    for (int nn_i = 1; nn_i < max_nn; ++nn_i) {
        if (radius < 0.0 || input_nn_dist[pc_id][nn_i] < radius * radius) {
            int nn_pc_id = input_nn_ind[pc_id][nn_i];
            float3 pos = make_float3(input_pc[nn_pc_id][0], input_pc[nn_pc_id][1], input_pc[nn_pc_id][2]);
            pos -= pc_mean;
            cov_x1.x += pos.x * pos.x; cov_x1.y += pos.x * pos.y; cov_x1.z += pos.x * pos.z;
            cov_x2.x += pos.y * pos.x; cov_x2.y += pos.y * pos.y; cov_x2.z += pos.y * pos.z;
            cov_x3.x += pos.z * pos.x; cov_x3.y += pos.z * pos.y; cov_x3.z += pos.z * pos.z;
        } else break;
    }

    float4 eigv = sym3eig(cov_x1, cov_x2, cov_x3);
    output_normal[pc_id][0] = eigv.x;
    output_normal[pc_id][1] = eigv.y;
    output_normal[pc_id][2] = eigv.z;
}

torch::Tensor estimateNormalsKNN(torch::Tensor xyz, torch::Tensor knnDist, torch::Tensor knnIndices, float maxRadius) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(knnDist);
    CHECK_CUDA(knnIndices);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_FLOAT(knnDist);
    long n_point = xyz.size(0);
    TORCH_CHECK(n_point == knnDist.size(0));
    TORCH_CHECK(n_point == knnIndices.size(0));
    TORCH_CHECK(knnDist.size(1) == knnIndices.size(1));

    torch::Tensor output_normal = torch::empty({n_point, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(128);
    dim3 dimGrid = dim3(div_up(n_point, dimBlock.x));
    estimate_normal_kernel<<<dimGrid, dimBlock>>>(
            xyz.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            knnDist.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            knnIndices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            knnDist.size(1), maxRadius,
            output_normal.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_normal;
}
