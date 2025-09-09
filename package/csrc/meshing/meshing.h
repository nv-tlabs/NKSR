#include <torch/extension.h>
#include <SparseFeatureIndexGrid.h>
#include <nanovdb/NanoVDB.h>
#include "../common/platform.h"

template<typename ScalarT>
_CPU_AND_GPU_CODE_ inline nanovdb::Vec4<ScalarT> sdf_interp(
        const nanovdb::Vec3<ScalarT> p1, const nanovdb::Vec3<ScalarT> p2, ScalarT valp1, ScalarT valp2) {

    if (std::abs(0.0f - valp1) < 1.0e-5f)
        return nanovdb::Vec4<ScalarT>(p1[0], p1[1], p1[2], 1.0);

    if (std::abs(0.0f - valp2) < 1.0e-5f)
        return nanovdb::Vec4<ScalarT>(p2[0], p2[1], p2[2], 0.0);

    if (std::abs(valp1 - valp2) < 1.0e-5f)
        return nanovdb::Vec4<ScalarT>(p1[0], p1[1], p1[2], 1.0);

    ScalarT w2 = (0.0 - valp1) / (valp2 - valp1);
    ScalarT w1 = 1 - w2;

    return nanovdb::Vec4<ScalarT>(
            p1[0] * w1 + p2[0] * w2,
            p1[1] * w1 + p2[1] * w2,
            p1[2] * w1 + p2[2] * w2, w1);
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline int get_cube_type(const ScalarT* sdf_vals) {
    int cube_type = 0;
    if (sdf_vals[0] < 0) cube_type |= 1;
    if (sdf_vals[1] < 0) cube_type |= 2;
    if (sdf_vals[2] < 0) cube_type |= 4;
    if (sdf_vals[3] < 0) cube_type |= 8;
    if (sdf_vals[4] < 0) cube_type |= 16;
    if (sdf_vals[5] < 0) cube_type |= 32;
    if (sdf_vals[6] < 0) cube_type |= 64;
    if (sdf_vals[7] < 0) cube_type |= 128;
    return cube_type;
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline void fill_vert_list(
        nanovdb::Vec4<ScalarT>* vert_list, int edge_config,
        nanovdb::Vec3<ScalarT>* points, ScalarT* sdf_vals) {
    if (edge_config & 1) vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
    if (edge_config & 2) vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
    if (edge_config & 4) vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
    if (edge_config & 8) vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
    if (edge_config & 16) vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
    if (edge_config & 32) vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
    if (edge_config & 64) vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
    if (edge_config & 128) vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
    if (edge_config & 256) vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
    if (edge_config & 512) vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
    if (edge_config & 1024) vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
    if (edge_config & 2048) vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3], sdf_vals[7]);
}

//template <typename BufferType>
//std::shared_ptr<SparseFeatureIndexGrid<BufferType>> buildFlattenedGrid(
//        const std::shared_ptr<SparseFeatureIndexGrid<BufferType>>& thisGrid,
//        const std::shared_ptr<SparseFeatureIndexGrid<BufferType>>& childGrid,
//        bool conforming);
//
//template <typename BufferType>
//std::shared_ptr<SparseFeatureIndexGrid<BufferType>> buildJointDualGrid(
//        const std::vector<std::shared_ptr<SparseFeatureIndexGrid<BufferType>>>& primalGrids);

template <typename BufferType>
torch::Tensor primalCubeGraph(const SparseFeatureIndexGrid<BufferType> &primal,
                              const SparseFeatureIndexGrid<BufferType> &dual,
                              unsigned nThreadsX = 128);

template <typename BufferType>
torch::Tensor dualCubeGraph(const std::vector<std::shared_ptr<SparseFeatureIndexGrid<BufferType>>>& primalGrids,
                            const SparseFeatureIndexGrid<BufferType> &corner,
                            unsigned nThreadsX = 128);


std::vector<torch::Tensor> MarchingCubesCPU(
        const torch::Tensor& cubeCornerInds,
        const torch::Tensor& cornerPos,
        const torch::Tensor& cornerValue);

std::vector<torch::Tensor> MarchingCubesCUDA(
        const torch::Tensor& cubeCornerInds,
        const torch::Tensor& cornerPos,
        const torch::Tensor& cornerValue);
