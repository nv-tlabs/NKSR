#include "keval.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>

template <>
void dispatchQgBuilding(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                        const fvdb::VoxelCoordTransform& transform,
                        const torch::Tensor& ptsPos,
                        const torch::Tensor& ptsKernel,
                        const torch::Tensor& gridKernel,
                        const torch::Tensor& gradKernelPts,
                        torch::Tensor& outIndexer,
                        torch::Tensor& outMatrix,
                        unsigned nThreadsX) {
    throw std::runtime_error("CPU version of QG build is not implemented!");
}

template <>
void dispatchQgBuildingBackward(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                                const fvdb::VoxelCoordTransform& transform,
                                const torch::Tensor& ptsPos,
                                const torch::Tensor& ptsKernel,
                                const torch::Tensor& gridKernel,
                                const torch::Tensor& gradKernelPts,
                                const torch::Tensor& gradOutMatrix,
                                torch::Tensor& gradPtsKernel,
                                torch::Tensor& gradGridKernel,
                                torch::Tensor& gradGradKernelPts,
                                unsigned nThreadsX) {
    throw std::runtime_error("CPU version of QG build is not implemented!");
}
