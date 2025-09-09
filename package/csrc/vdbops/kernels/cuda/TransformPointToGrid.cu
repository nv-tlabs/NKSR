#include "Kernels.h"

#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


template <typename ScalarT>
__global__ void transformPointsToGrid(const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptCoordsAcc,
                                      torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outCoordsAcc,
                                      const fvdb::VoxelCoordTransform tx) {

    const int32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= ptCoordsAcc.size(0)) {
        return;
    }

    const nanovdb::Vec3<ScalarT> wci = tx.apply(ptCoordsAcc[i][0],
                                                ptCoordsAcc[i][1],
                                                ptCoordsAcc[i][2]);
    outCoordsAcc[i][0] = wci[0];
    outCoordsAcc[i][1] = wci[1];
    outCoordsAcc[i][2] = wci[2];

}


template <typename ScalarT>
__global__ void transformPointsToGridGrad(const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outGradAcc,
                                          torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outGradInAcc,
                                          const fvdb::VoxelCoordTransform tx) {

    const int32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= outGradAcc.size(0)) {
        return;
    }

    const nanovdb::Vec3<ScalarT> wci = tx.applyGrad(outGradAcc[i][0],
                                                    outGradAcc[i][1],
                                                    outGradAcc[i][2]);
    outGradInAcc[i][0] = wci[0] * outGradAcc[i][0];
    outGradInAcc[i][1] = wci[1] * outGradAcc[i][1];
    outGradInAcc[i][2] = wci[2] * outGradAcc[i][2];
}


template <typename ScalarT>
__global__ void invTransformPointsToGrid(const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptCoordsAcc,
                                         torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outCoordsAcc,
                                         fvdb::VoxelCoordTransform tx) {
    const int32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= ptCoordsAcc.size(0)) {
        return;
    }

    const nanovdb::Vec3<ScalarT> wci = tx.applyInv(ptCoordsAcc[i][0],
                                                    ptCoordsAcc[i][1],
                                                    ptCoordsAcc[i][2]);
    outCoordsAcc[i][0] = wci[0];
    outCoordsAcc[i][1] = wci[1];
    outCoordsAcc[i][2] = wci[2];
}


template <typename ScalarT>
__global__ void invTransformPointsToGridGrad(const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outGradAcc,
                                             torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outGradInAcc,
                                             const fvdb::VoxelCoordTransform tx) {

    const int32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= outGradAcc.size(0)) {
        return;
    }

    const nanovdb::Vec3<ScalarT> wci = tx.applyInvGrad(outGradAcc[i][0],
                                                       outGradAcc[i][1],
                                                       outGradAcc[i][2]);
    outGradInAcc[i][0] = wci[0] * outGradAcc[i][0];
    outGradInAcc[i][1] = wci[1] * outGradAcc[i][1];
    outGradInAcc[i][2] = wci[2] * outGradAcc[i][2];
}




namespace fvdb {

template <>
void dispatchTransformPointsToGrid<PytorchDeviceBuffer>(const torch::Tensor& points,
                                                        const VoxelCoordTransform& transform,
                                                        torch::Tensor& outCoords,
                                                        unsigned nThreadsX) {
    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "transformPointsToGrid", ([&] {
            transformPointsToGrid<scalar_t><<<nblocks, nthreads>>>(
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outCoords.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchInvTransformPointsToGrid<PytorchDeviceBuffer>(const torch::Tensor& points,
                                                           const VoxelCoordTransform& transform,
                                                           torch::Tensor& outCoords,
                                                           unsigned nThreadsX) {
    const int64_t PCOUNT = points.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        points.scalar_type(), "invTransformPointsToGrid", ([&] {
            invTransformPointsToGrid<scalar_t><<<nblocks, nthreads>>>(
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outCoords.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchTransformPointsToGridGrad<PytorchDeviceBuffer>(const torch::Tensor& gradOut,
                                                            const VoxelCoordTransform& transform,
                                                            torch::Tensor& outGradIn,
                                                            unsigned nThreadsX) {
    const int64_t PCOUNT = gradOut.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gradOut.scalar_type(), "transformPointsToGridGrad", ([&] {
            transformPointsToGridGrad<scalar_t><<<nblocks, nthreads>>>(
                gradOut.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outGradIn.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <>
void dispatchInvTransformPointsToGridGrad<PytorchDeviceBuffer>(const torch::Tensor& gradOut,
                                                               const VoxelCoordTransform& transform,
                                                               torch::Tensor& outGradIn,
                                                               unsigned nThreadsX) {
    const int64_t PCOUNT = gradOut.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gradOut.scalar_type(), "invTransformPointsToGridGrad", ([&] {
            invTransformPointsToGridGrad<scalar_t><<<nblocks, nthreads>>>(
                gradOut.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outGradIn.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                transform);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb