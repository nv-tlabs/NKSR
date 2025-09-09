#include "Kernels.h"

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/TrilinearInterpolationWithGradIterator.h"
#include "utils/PytorchDeviceBuffer.h"


template <typename Dtype>
__global__ void sampleGridTrilinear(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                    const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                                    const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> gridData,
                                    torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outFeatures,
                                    fvdb::VoxelCoordTransform transform) {
    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t fj = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pi >= points.size(0) || fj >= gridData.size(1)) {
        return;
    }

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0],
                                                     points[pi][1],
                                                     points[pi][2]);

    #pragma unroll
    for (auto it = fvdb::TrilinearInterpolationIterator<Dtype>(xyz); it.isValid(); ++it) {
        const Dtype wTrilinear = it->second;
        const nanovdb::Coord ijk = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1;
            outFeatures[pi][fj] += wTrilinear * gridData[indexIjk][fj];
        }
    }
}


template <typename Dtype>
__global__ void sampleGridTrilinearWithGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                            const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                                            const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> gridData,
                                            torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outFeatures,
                                            torch::PackedTensorAccessor32<Dtype, 3, torch::RestrictPtrTraits> outGradFeatures,
                                            fvdb::VoxelCoordTransform transform) {
    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t fj = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pi >= points.size(0) || fj >= gridData.size(1)) {
        return;
    }

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0],
                                                     points[pi][1],
                                                     points[pi][2]);

#pragma unroll
    for (auto it = fvdb::TrilinearInterpolationWithGradIterator<Dtype>(xyz); it.isValid(); ++it) {
        const nanovdb::Vec4<Dtype> wXYZ = it->second;
        const nanovdb::Coord ijk = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1;
            outFeatures[pi][fj] += wXYZ[0] * gridData[indexIjk][fj];
            auto scale = transform.template scale<Dtype>();
#pragma unroll
            for (int dim = 0; dim < 3; ++ dim) {
                outGradFeatures[pi][fj][dim] += wXYZ[dim + 1] * gridData[indexIjk][fj] * scale;
            }
        }
    }
}


template <typename Dtype>
__global__ void sampleGridTrilinearWithGradGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                                const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> points,
                                                const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> gradOutFeatures,
                                                const torch::PackedTensorAccessor32<Dtype, 3, torch::RestrictPtrTraits> gradOutGradFeatures,
                                                torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outGridData,
                                                fvdb::VoxelCoordTransform transform) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t fj = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pi >= points.size(0) || fj >= gradOutFeatures.size(1)) {
        return;
    }

    auto gridAcc = gpuGrid->getAccessor();

    const nanovdb::Vec3<Dtype> xyz = transform.apply(points[pi][0], points[pi][1], points[pi][2]);

#pragma unroll
    for (auto it = fvdb::TrilinearInterpolationWithGradIterator<Dtype>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk = gridAcc.getValue(it->first) - 1;
            Dtype addValue = it->second[0] * gradOutFeatures[pi][fj];
            auto scale = transform.template scale<Dtype>();
#pragma unroll
            for (int dim = 0; dim < 3; ++dim) {
                addValue += it->second[dim + 1] * gradOutGradFeatures[pi][fj][dim] * scale;
            }
            gpuAtomicAddNoReturn(&outGridData[indexIjk][fj], addValue);
        }
    }
}




namespace fvdb {

template<>
void dispatchSampleGridTrilinear<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                      const torch::Tensor& points,
                                                      const torch::Tensor& gridData,
                                                      const VoxelCoordTransform& transform,
                                                      torch::Tensor& outFeatures,
                                                      unsigned nThreadsX, unsigned nThreadsY) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    const int64_t DCOUNT = gridData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "sampleGridTrilinear", ([&] {
            sampleGridTrilinear<scalar_t><<<nblocks, nthreads>>>(
                gpuGrid,
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outFeatures.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchSampleGridTrilinearWithGrad<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                              const torch::Tensor& points,
                                                              const torch::Tensor& gridData,
                                                              const VoxelCoordTransform& transform,
                                                              torch::Tensor& outFeatures,
                                                              torch::Tensor& outGradFeatures,
                                                              unsigned nThreadsX, unsigned nThreadsY) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    const int64_t DCOUNT = gridData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "sampleGridTrilinearWithGrad", ([&] {
            sampleGridTrilinearWithGrad<scalar_t><<<nblocks, nthreads>>>(
                gpuGrid,
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outFeatures.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outGradFeatures.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchSampleGridTrilinearWithGradGrad<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                                  const torch::Tensor& points,
                                                                  const torch::Tensor& gradOutFeatures,
                                                                  const torch::Tensor& gradOutGradFeatures,
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

    const int64_t DCOUNT = gradOutFeatures.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "sampleGridTrilinearWithGradGrad", ([&] {
        sampleGridTrilinearWithGradGrad<scalar_t><<<nblocks, nthreads>>>(
            gpuGrid,
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gradOutFeatures.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gradOutGradFeatures.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            outGridData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb