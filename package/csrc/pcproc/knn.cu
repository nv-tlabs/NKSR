#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

#include "cuda_kdtree.cuh"
#include "pcproc.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

std::vector<torch::Tensor> nearestNeighbours(torch::Tensor xyz, int knn) {
    CHECK_CUDA(xyz);
    CHECK_IS_FLOAT(xyz);
    TORCH_CHECK(xyz.size(0) >= knn, "knn is too small compared to the size of point cloud!");

    torch::Tensor strided_xyz = xyz;
    torch::Device device = xyz.device();
    long n_point = xyz.size(0);

    if (strided_xyz.stride(0) != 4) {
        strided_xyz = torch::zeros({n_point, 4}, torch::dtype(torch::kFloat32).device(device));
        strided_xyz.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}, xyz);
    }

    auto* knn_index = new tinyflann::KDTreeCuda3dIndex<tinyflann::CudaL2>(strided_xyz.data_ptr<float>(), n_point);
    knn_index->buildIndex();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor dist = torch::empty(n_point * knn, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor indices = torch::empty(n_point * knn, torch::dtype(torch::kInt32).device(device));

    knn_index->knnSearch(strided_xyz.data_ptr<float>(), n_point, 4, indices.data_ptr<int>(),
                         dist.data_ptr<float>(), knn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    delete knn_index;

    return {dist.reshape({n_point, knn}), indices.reshape({n_point, knn})};
}
