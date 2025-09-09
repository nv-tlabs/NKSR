#include "Kernels.h"

#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


template <typename Dtype>
__global__ void pointsInGrid(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                             const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                             torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> outMask,
                             fvdb::VoxelCoordTransform transform) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pi >= points.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0],
                                                     points[pi][1],
                                                     points[pi][2]);

    const nanovdb::Coord vox = xyz.round();
    if (primalAcc.isActive(vox)) {
        outMask[pi] = true;
    } else {
        outMask[pi] = false;
    }
}


namespace fvdb {

template <>
void dispatchPointsInGrid<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                               const VoxelCoordTransform& transform,
                                               const torch::Tensor& points, torch::Tensor& outMask,
                                               unsigned nThreadsX) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "pointsInGrid", [&]() {
        pointsInGrid<scalar_t><<<nblocks, nthreads>>>(
            gpuGrid,
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            outMask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            transform);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb
