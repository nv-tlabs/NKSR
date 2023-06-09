#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "../common/kdtree_cuda.cuh"
#include "../common/cutil_math.h"

using DataAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using SDFAccessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;

class ThrustAllocator {
public:
    typedef char value_type;

    char* allocate(std::ptrdiff_t size) {
        return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
    }

    void deallocate(char* p, size_t size) {
        c10::cuda::CUDACachingAllocator::raw_delete(p);
    }
};

// https://arxiv.org/pdf/2203.09167.pdf
__global__ static void ComputeIMLSKernel(size_t num_samples, int num_votes,
                                         const DataAccessor ref_xyz, const DataAccessor ref_normals,
                                         const int* __restrict__ knn_index,
                                         const DataAccessor query_xyz,
                                         float stdv,
                                         SDFAccessor sdf_val, DataAccessor sdf_grad,
                                         bool compute_grad) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) {
        return;
    }

    float3 sample_pos = make_float3(query_xyz[sample_id][0], query_xyz[sample_id][1], query_xyz[sample_id][2]);

    float sdf = 0.0;
    float3 d_sdf = make_float3(0.0, 0.0, 0.0);
    float weight_sum = 0.0;

    // Compute min weight for subtraction.
    float min_exp_val = 1e8;
    for (int vote_i = 0; vote_i < num_votes; ++vote_i) {
        int cur_ind = knn_index[sample_id * num_votes + vote_i];
        float3 nb_pos = make_float3(ref_xyz[cur_ind][0], ref_xyz[cur_ind][1], ref_xyz[cur_ind][2]);

        // n_k -> nb_normal, p_k -> nb_pos, x -> sample_pos
        float3 ray_vec = sample_pos - nb_pos;
        float exp_val = dot(ray_vec, ray_vec) / (stdv * stdv);
        min_exp_val = min(min_exp_val, exp_val);
    }

    for (int vote_i = 0; vote_i < num_votes; ++vote_i) {
        int cur_ind = knn_index[sample_id * num_votes + vote_i];
        float3 nb_pos = make_float3(ref_xyz[cur_ind][0], ref_xyz[cur_ind][1], ref_xyz[cur_ind][2]);
        float3 nb_normal = make_float3(ref_normals[cur_ind][0], ref_normals[cur_ind][1], ref_normals[cur_ind][2]);

        // n_k -> nb_normal, p_k -> nb_pos, x -> sample_pos
        float3 ray_vec = sample_pos - nb_pos;
        float d = dot(nb_normal, ray_vec);

        float exp_val = dot(ray_vec, ray_vec) / (stdv * stdv);
        float w = exp(-exp_val + min_exp_val);

        weight_sum += w;
        sdf += d * w;
        if (compute_grad) {
            d_sdf += nb_normal * w;
        }
    }

    sdf_val[sample_id] = sdf / weight_sum;
    if (compute_grad) {
        sdf_grad[sample_id][0] = d_sdf.x / weight_sum;
        sdf_grad[sample_id][1] = d_sdf.y / weight_sum;
        sdf_grad[sample_id][2] = d_sdf.z / weight_sum;
    }

}

__global__ static void ComputeSDFKernel(size_t num_samples, int num_votes,
                                        const DataAccessor ref_xyz, const DataAccessor ref_normals,
                                        const SDFAccessor ref_std,
                                        const int* __restrict__ knn_index,
                                        const DataAccessor query_xyz,
                                        float stdv,
                                        SDFAccessor sdf_val, DataAccessor sdf_grad,
                                        bool compute_grad) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) {
        return;
    }

    float3 sample_pos = make_float3(query_xyz[sample_id][0], query_xyz[sample_id][1], query_xyz[sample_id][2]);

    float sdf;
    float3 d_sdf;
    int num_pos = 0;
    for (int vote_i = 0; vote_i < num_votes; ++vote_i) {
        int cur_ind = knn_index[sample_id * num_votes + vote_i];
        float3 nb_pos = make_float3(ref_xyz[cur_ind][0], ref_xyz[cur_ind][1], ref_xyz[cur_ind][2]);
        float3 nb_normal = make_float3(ref_normals[cur_ind][0], ref_normals[cur_ind][1], ref_normals[cur_ind][2]);
        float3 ray_vec = sample_pos - nb_pos;

        float d = dot(nb_normal, ray_vec);
        if (vote_i == 0) {
            float ray_vec_len = length(ray_vec);
            if (ray_vec_len < stdv * ref_std[cur_ind]) {
                sdf = abs(d);
                if (compute_grad) {
                    d_sdf = d > 0 ? nb_normal : -nb_normal;
                }
            } else {
                sdf = ray_vec_len;
                if (compute_grad) {
                    d_sdf = ray_vec / sdf;
                }
            }
        }
        if (d > 0) { num_pos += 1; }
    }

    if (num_pos <= num_votes / 2) {
        sdf_val[sample_id] = -sdf;
        if (compute_grad) {
            sdf_grad[sample_id][0] = -d_sdf.x;
            sdf_grad[sample_id][1] = -d_sdf.y;
            sdf_grad[sample_id][2] = -d_sdf.z;
        }
    } else {
        sdf_val[sample_id] = sdf;
        if (compute_grad) {
            sdf_grad[sample_id][0] = d_sdf.x;
            sdf_grad[sample_id][1] = d_sdf.y;
            sdf_grad[sample_id][2] = d_sdf.z;
        }
    }
}

std::vector<torch::Tensor> sdf_from_points(
        const torch::Tensor& queries, const torch::Tensor& ref_xyz, const torch::Tensor& ref_normal,
        int nb_points, float stdv, bool compute_grad, bool imls, int adaptive_knn) {

    CHECK_CUDA(queries); CHECK_IS_FLOAT(queries)
    CHECK_CUDA(ref_xyz);
    CHECK_CUDA(ref_normal);

    // Index requires reference to have stride 4.
    torch::Tensor strided_ref = ref_xyz;
    torch::Device device = queries.device();
    if (ref_xyz.stride(0) != 4) {
        strided_ref = torch::zeros({strided_ref.size(0), 4}, torch::dtype(torch::kFloat32).device(device));
        strided_ref.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}, ref_xyz);
    }

    // Note: if needed, should refer to this:
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//    ThrustAllocator allocator;
//    auto policy = thrust::cuda::par(allocator).on(stream);

    // Build KDTree based on reference point cloud
    size_t n_ref = strided_ref.size(0);
    auto* knn_index = new tinyflann::KDTreeCuda3dIndex<tinyflann::CudaL2>(strided_ref.data_ptr<float>(), n_ref);
    knn_index->buildIndex();

    // If determine threshold adaptively, compute the thresholds here (re-using the kd-tree)
    torch::Tensor ref_std = torch::ones(n_ref, torch::dtype(torch::kFloat32).device(device));
    if (adaptive_knn > 0) {
        torch::Tensor ref_dist = torch::empty(n_ref * adaptive_knn, torch::dtype(torch::kFloat32).device(device));
        torch::Tensor ref_indices = torch::empty(n_ref * adaptive_knn, torch::dtype(torch::kInt32).device(device));
        knn_index->knnSearch(strided_ref.data_ptr<float>(), n_ref, strided_ref.stride(0), ref_indices.data_ptr<int>(),
                            ref_dist.data_ptr<float>(), adaptive_knn);
        ref_dist.sqrt_();
        ref_std = ref_dist.view({(int) n_ref, adaptive_knn}).mean(1);
    }

    // Compute for each point its nearest N neighbours.
    size_t n_query = queries.size(0);
    torch::Tensor dist = torch::empty(n_query * nb_points, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor indices = torch::empty(n_query * nb_points, torch::dtype(torch::kInt32).device(device));
    knn_index->knnSearch(queries.data_ptr<float>(), n_query, queries.stride(0), indices.data_ptr<int>(),
                        dist.data_ptr<float>(), nb_points);
    delete knn_index;

    // Compute sdf value and the unit gradient.
    torch::Tensor sdf = torch::zeros(n_query, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor grad_sdf = torch::zeros({0, 3}, torch::dtype(torch::kFloat32).device(device));
    if (compute_grad) {
        grad_sdf = torch::zeros({(int) n_query, 3}, torch::dtype(torch::kFloat32).device(device));
    }

    if (imls) {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((n_query + dimBlock.x - 1) / dimBlock.x);
        ComputeIMLSKernel<<<dimGrid, dimBlock>>>(n_query, nb_points,
                                                ref_xyz.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                ref_normal.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                 indices.data_ptr<int>(),
                                                queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                stdv,
                                                sdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                grad_sdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                compute_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((n_query + dimBlock.x - 1) / dimBlock.x);
        ComputeSDFKernel<<<dimGrid, dimBlock>>>(n_query, nb_points,
                                                ref_xyz.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                ref_normal.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                ref_std.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                indices.data_ptr<int>(),
                                                queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                stdv,
                                                sdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                grad_sdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                compute_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (compute_grad) {
        return {sdf, grad_sdf};
    } else {
        return {sdf};
    }
}
