#include "Kernels.h"

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


template <typename Dtype>
__global__ void splatIntoGridTrilinear(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                       const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                                       const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> pointsData,
                                       torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outGridData,
                                       fvdb::VoxelCoordTransform transform) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t fj = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pi >= points.size(0) || fj >= pointsData.size(1)) {
        return;
    }

    auto gridAcc = gpuGrid->getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0], points[pi][1], points[pi][2]);

    #pragma unroll
    for (auto it = fvdb::TrilinearInterpolationIterator<Dtype>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk = gridAcc.getValue(it->first) - 1;
            const Dtype addValue = it->second * pointsData[pi][fj];
            gpuAtomicAddNoReturn(&outGridData[indexIjk][fj], addValue);
        }
    }
}


template <typename Dtype>
__global__ void splatIntoGridTrilinearWithCounts(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                                 const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                                                 const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> pointsData,
                                                 torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outGridData,
                                                 torch::PackedTensorAccessor32<Dtype, 1, torch::RestrictPtrTraits> outGridCounts,
                                                 fvdb::VoxelCoordTransform transform) {
    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t fj = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pi >= points.size(0) || fj >= pointsData.size(1)) {
        return;
    }

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0], points[pi][1], points[pi][2]);

    #pragma unroll
    for (auto it = fvdb::TrilinearInterpolationIterator<Dtype>(xyz); it.isValid(); it++) {
        if (gridAcc.isActive(it->first)) {
            const uint64_t offset = gridAcc.getValue(it->first) - 1;
            const Dtype addValue = it->second * pointsData[pi][fj];
            gpuAtomicAddNoReturn(&outGridData[offset][fj], addValue);
            gpuAtomicAddNoReturn(&outGridCounts[offset], static_cast<Dtype>(1));
        }
    }
}





namespace fvdb {

template <>
void dispatchSplatIntoGridTrilinear<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                         const torch::Tensor& points,
                                                         const torch::Tensor& pointsData,
                                                         const VoxelCoordTransform& transform,
                                                         torch::Tensor& outGridData,
                                                         unsigned nThreadsX, unsigned nThreadsY) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    const int64_t DCOUNT = pointsData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "splatIntoGridTrilinear", ([&] {
            splatIntoGridTrilinear<scalar_t><<<nblocks, nthreads>>>(
                gpuGrid,
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                pointsData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outGridData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchSplatIntoGridTrilinearWithCounts<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                                   const torch::Tensor& points,
                                                                   const torch::Tensor& pointPrimalData,
                                                                   const VoxelCoordTransform& transform,
                                                                   torch::Tensor& outPrimalData,
                                                                   torch::Tensor& outCounts,
                                                                   unsigned nThreadsX, unsigned nThreadsY) {
    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    const int64_t DCOUNT = pointPrimalData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "splatIntoGridTrilinearWithCounts", ([&] {
            splatIntoGridTrilinearWithCounts<scalar_t><<<nblocks, nthreads>>>(
                gpuGrid,
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                pointPrimalData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outPrimalData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outCounts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb