#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
#include <utils/cuda/Math.cuh>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;


/**
 * RHS Evaluation (only with gradient)
 */

template <typename ScalarT>
__global__ void rhsEvaluation(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> pts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelPts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsData,
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> outRhs) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= pts.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi][0], pts[pi][1], pts[pi][2]);
    nanovdb::Vec3<ScalarT> pData(ptsData[pi][0], ptsData[pi][1], ptsData[pi][2]);

    // For each point, iterate through all its neighbours.
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1;

        // Kernel (and gradient) evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, pi, transform.scale<ScalarT>(),
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                true, kiv, gradKiv, bk, dk, db);

        ScalarT res = pData.dot(gradKiv);
        gpuAtomicAddNoReturn(&outRhs[offset], res);
    }
}

template <>
void dispatchRhsEvaluation(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                           const fvdb::VoxelCoordTransform& transform,
                           const torch::Tensor& pts,
                           const torch::Tensor& ptsKernel,
                           const torch::Tensor& gridKernel,
                           const torch::Tensor& gradKernelPts,
                           const torch::Tensor& ptsData,
                           torch::Tensor& outRhs,
                           unsigned nThreadsX) {
    const auto* gridGrid = grid.template deviceGrid<nanovdb::ValueIndex>();
    if (!gridGrid) {
        throw std::runtime_error("cannot obtain grid in rhs evaluation!");
    }

    const int64_t PCOUNT = pts.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "rhsEvaluation", [&]() {
        rhsEvaluation<scalar_t><<<nblocks, nthreads>>>(
                gridGrid, transform,
                pts.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                ptsData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outRhs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ScalarT>
__global__ void rhsEvaluationBackward(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> pts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelPts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsData,
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradOutRhs,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradPtsKernel,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradGridKernel,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradGradKernelPts,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradPtsData) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= pts.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi]);
    nanovdb::Vec3<ScalarT> pData(ptsData[pi][0], ptsData[pi][1], ptsData[pi][2]);

    // For each point, iterate through all its neighbours.
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1;

        // Kernel (and gradient) evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, pi, transform.scale<ScalarT>(),
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                true, kiv, gradKiv, bk, dk, db);

        auto dummyAcc2 = ptsData;
        auto dummyAcc1 = gradOutRhs;

        // Backward
        kernel_grad_evaluation_bwd<ScalarT, true>(
                offset, pi,
                ptsKernel, gridKernel, gradKernelPts,
                gradOutRhs, dummyAcc2, true, 1.0,
                gradPtsKernel, gradGridKernel, dummyAcc1, gradGradKernelPts,
                offset, 0.0, pData,
                kiv, gradKiv, bk, dk, db);
        for (int dim = 0; dim < 3; ++dim) {
            gradPtsData[pi][dim] += gradOutRhs[offset] * gradKiv[dim];
        }
    }
}

template <>
void dispatchRhsEvaluationBackward(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                                   const fvdb::VoxelCoordTransform& transform,
                                   const torch::Tensor& pts,
                                   const torch::Tensor& ptsKernel,
                                   const torch::Tensor& gridKernel,
                                   const torch::Tensor& gradKernelPts,
                                   const torch::Tensor& ptsData,
                                   const torch::Tensor& gradOutRhs,
                                   torch::Tensor& gradPtsKernel,
                                   torch::Tensor& gradGridKernel,
                                   torch::Tensor& gradGradKernelPts,
                                   torch::Tensor& gradPtsData,
                                   unsigned nThreadsX) {
    const auto* gridGrid = grid.deviceGrid<nanovdb::ValueIndex>();

    const int64_t PCOUNT = pts.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "rhsEvaluationBackward", [&]() {
        rhsEvaluationBackward<scalar_t><<<nblocks, nthreads>>>(
                gridGrid, transform,
                pts.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                ptsData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradOutRhs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                gradPtsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradGridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradGradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                gradPtsData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
