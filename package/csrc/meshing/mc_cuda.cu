#include "mc_data.h"
#include "meshing.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")
#define CHECK_IS_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be a long tensor")

__global__ static void classify_voxels(const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> cube_corner_inds,
                                       const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> corner_value,
                                       torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vertex_counts) {
    const int64_t num_cubes = cube_corner_inds.size(0);
    const int64_t cube_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cube_idx >= num_cubes) {
        return;
    }

    float sdf_vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sdf_vals[i] = corner_value[cube_corner_inds[cube_idx][i]];
    }

    // Find triangle config.
    int cube_type = get_cube_type(sdf_vals);
    vertex_counts[cube_idx] = numVertsTable[cube_type];
}

__global__ static void meshing_cube(const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> cube_corner_inds,
                                    const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> corner_value,
                                    const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> corner_pos,
                                    torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> triangles,
                                    torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> vert_ids,
                                    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> count_csum) {
    const int64_t num_cubes = cube_corner_inds.size(0);
    const int64_t cube_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cube_idx >= num_cubes) {
        return;
    }

    float sdf_vals[8];
    int64_t point_ids[8];
    nanovdb::Vec3f points[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        point_ids[i] = cube_corner_inds[cube_idx][i];
        sdf_vals[i] = corner_value[point_ids[i]];
        for (int d = 0; d < 3; ++d) {
            points[i][d] = corner_pos[point_ids[i]][d];
        }
    }

    // Find triangle config.
    int cubeType = get_cube_type(sdf_vals);
    int edgeConfig = edgeTable[cubeType];
    if (edgeConfig == 0) return;

    nanovdb::Vec4f vert_list[12];
    fill_vert_list(vert_list, edgeConfig, points, sdf_vals);

    // Write triangles to array.
    for (int i = 0; triangleTable[cubeType][i] != -1; i += 3) {
        int64_t triangle_id = count_csum[cube_idx] / 3 + i / 3;
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            int64_t vlid = triangleTable[cubeType][i + vi];
            for (int d = 0; d < 3; ++d) {
                triangles[triangle_id][vi][d] = vert_list[vlid][d];
            }
            int64_t vid0 = point_ids[e2iTable[vlid][0]];
            int64_t vid1 = point_ids[e2iTable[vlid][1]];
            if (vid0 < vid1) {
                int64_t t = vid1; vid1 = vid0; vid0 = t;
            }
            vert_ids[triangle_id][vi][0] = vid0;
            vert_ids[triangle_id][vi][1] = vid1;
        }
    }
}

std::vector<torch::Tensor> MarchingCubesCUDA(
        const torch::Tensor &cubeCornerInds,    // (M, 8) long
        const torch::Tensor &cornerPos,         // (N, 3) float
        const torch::Tensor &cornerValue) {     // (N, )  float

    CHECK_CUDA(cubeCornerInds); CHECK_IS_LONG(cubeCornerInds);
    CHECK_CUDA(cornerPos); CHECK_IS_FLOAT(cornerPos);
    CHECK_CUDA(cornerValue); CHECK_IS_FLOAT(cornerValue);

    int64_t numCubes = cubeCornerInds.size(0);

    dim3 dimBlock = dim3(256);
    dim3 dimGrid = dim3((numCubes + dimBlock.x - 1) / dimBlock.x);

    const auto& opt = torch::TensorOptions().device(cubeCornerInds.device());
    torch::Tensor vertexCounts = torch::empty(numCubes, opt.dtype(torch::kLong));

    if (numCubes > 0) {
        classify_voxels<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cubeCornerInds.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                cornerValue.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                vertexCounts.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // cumsum to determine starting position.
    //    We do not perform compaction, because we are already fast.
    torch::Tensor countCsum = torch::cumsum(vertexCounts, 0);
    int64_t n_triangles = 0;
    if (numCubes > 0) {
        n_triangles = countCsum[-1].item<int64_t>() / 3;
        countCsum = torch::roll(countCsum, torch::IntList(1));
        countCsum[0] = 0;
    }

    // Generate triangles
    torch::Tensor triangles = torch::empty({n_triangles, 3, 3}, opt.dtype(torch::kFloat32));
    torch::Tensor vert_ids = torch::empty({n_triangles, 3, 2}, opt.dtype(torch::kLong));

    if (n_triangles > 0) {
        meshing_cube<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cubeCornerInds.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                cornerValue.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                cornerPos.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                triangles.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
                vert_ids.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
                countCsum.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Flatten
    triangles = triangles.view({-1, 3});
    vert_ids = vert_ids.view({-1, 2});

    // Filter degenerated vertices (not necessary)
//    torch::Tensor degMask = vert_ids.index(
//            {torch::indexing::Ellipsis, 0}) != vert_ids.index({torch::indexing::Ellipsis, 1});
//    vert_ids = vert_ids.index({degMask});
//    triangles = triangles.index({degMask});

    // Merge triangles by detecting the same vertex position.
    const auto &unqRet = torch::unique_dim(vert_ids, 0, false, true);
    const torch::Tensor& unqVertIdx = std::get<0>(unqRet);
    const torch::Tensor& unqTriangles = std::get<1>(unqRet);

    torch::Tensor vertices = torch::zeros({unqVertIdx.size(0), 3}, opt.dtype(torch::kFloat32));
    vertices.index_put_({unqTriangles}, triangles);

    return {vertices, unqTriangles.view({-1, 3}), unqVertIdx};
}
