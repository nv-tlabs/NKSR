#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
#include <utils/cuda/Math.cuh>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;

// Get rid of fp64 operations by re-writing them
template <typename ScalarT>
__forceinline__ __device__ nanovdb::Coord roundVec(const nanovdb::Vec3<ScalarT>& vec) {
    return vec.round();
}

template <>
__forceinline__ __device__ nanovdb::Coord roundVec(const nanovdb::Vec3<float>& vec) {
    return nanovdb::Coord(
            (int32_t) lroundf(vec[0]),
            (int32_t) lroundf(vec[1]),
            (int32_t) lroundf(vec[2]));
}

/**
 * Matrix Building
 */

template <typename ScalarT, typename IndexT>
__global__ void matrixBuilding(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridI,
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridJ,
        fvdb::VoxelCoordTransform transformI,
        fvdb::VoxelCoordTransform transformJ,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsPos,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernelI,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernelJ,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> iKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> jKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradPtsKernelPosI,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradPtsKernelPosJ,
        const torch::PackedTensorAccessor32<IndexT, 2, torch::RestrictPtrTraits> indexMap,   // long Tensor (I, 125)
        bool grad,          // Build GTG or QTQ
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> outMatrix) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t iti = (blockIdx.y * blockDim.y) + threadIdx.y;

    using NN = NNIterator<3, ScalarT>;
    if (pi >= ptsPos.size(0) || iti >= NN::total()) {
        return;
    }

    auto iAcc = gridI->getAccessor();
    const nanovdb::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPos[pi]);
    auto it = NN(roundVec(piLocal), iti);

    if (!iAcc.isActive(*it)) {
        return;
    }
    const int64_t offsetI = iAcc.getValue(*it) - 1;

    // Evaluate kernel K(k, i)
    ScalarT kiF = 0.0, kiBk, kiDk;
    nanovdb::Vec3<ScalarT> gradKiF(0.0), kiDb(0.0);
    kernel_grad_evaluation_fwd(
            offsetI, pi, transformI.scale<ScalarT>(),
            piLocal[0] - (ScalarT) (*it)[0],
            piLocal[1] - (ScalarT) (*it)[1],
            piLocal[2] - (ScalarT) (*it)[2],
            ptsKernelI, iKernel, gradPtsKernelPosI,
            grad, kiF, gradKiF, kiBk, kiDk, kiDb);

    auto jAcc = gridJ->getAccessor();
    const nanovdb::Vec3<ScalarT> pjLocal = transformJ.apply<ScalarT>(ptsPos[pi]);

    // Iterate over index [j] (to be put into one kernel execution)
#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(roundVec(pjLocal)); jt.isValid(); ++jt) {
        if (!jAcc.isActive(*jt)) {
            continue;
        }
        const int64_t offsetJ = jAcc.getValue(*jt) - 1;

        // Evaluate kernel K(k, j)
        ScalarT kjF = 0.0, kjBk, kjDk;
        nanovdb::Vec3<ScalarT> gradKjF(0.0), kjDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, pi, transformJ.scale<ScalarT>(),
                pjLocal[0] - (ScalarT) (*jt)[0],
                pjLocal[1] - (ScalarT) (*jt)[1],
                pjLocal[2] - (ScalarT) (*jt)[2],
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                grad, kjF, gradKjF, kjBk, kjDk, kjDb);

        // Put K(k,i)*K(k,j) into Mat[offsetI, offsetJ], aka. -> outMatrix[outMatrixIdx]
        ScalarT outVal;
        if (!grad) { outVal = kiF * kjF; }
        else { outVal = gradKiF.template dot(gradKjF); }

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(it->asVec3s())));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        gpuAtomicAddNoReturn(&outMatrix[outMatrixIdx], outVal);
    }
}

template <>
void dispatchMatrixBuilding(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& gridI,
                            const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& gridJ,
                            const fvdb::VoxelCoordTransform& transformI,
                            const fvdb::VoxelCoordTransform& transformJ,
                            const torch::Tensor& ptsPos,
                            const torch::Tensor& ptsKernelI,
                            const torch::Tensor& ptsKernelJ,
                            const torch::Tensor& iKernel,
                            const torch::Tensor& jKernel,
                            const torch::Tensor& gradPtsKernelPosI,
                            const torch::Tensor& gradPtsKernelPosJ,
                            const torch::Tensor& indexMap,
                            bool grad,
                            torch::Tensor& outMatrix,
                            unsigned nThreadsX,
                            unsigned nThreadsY) {
    const auto* gridGridI = gridI.deviceGrid<nanovdb::ValueIndex>();
    const auto* gridGridJ = gridJ.deviceGrid<nanovdb::ValueIndex>();
    if (!gridGridI || !gridGridJ) {
        throw std::runtime_error("Failed to get pointer for nanovdb index grid");
    }

    const int64_t PCOUNT = ptsPos.size(0);
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, float>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    if (indexMap.scalar_type() == torch::kInt32) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuilding", [&]() {
            matrixBuilding<scalar_t, int><<<nblocks, nthreads>>>(
                    gridGridI, gridGridJ, transformI, transformJ,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    grad,
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        });
    } else if (indexMap.scalar_type() == torch::kLong) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuilding", [&]() {
            matrixBuilding<scalar_t, int64_t><<<nblocks, nthreads>>>(
                    gridGridI, gridGridJ, transformI, transformJ,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    grad,
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Scalar type of indexer is not supported!");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ScalarT, typename IndexT>
__global__ void matrixBuildingBackward(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridI,
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridJ,
        fvdb::VoxelCoordTransform transformI,
        fvdb::VoxelCoordTransform transformJ,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsPos,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernelI,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernelJ,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> iKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> jKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradPtsKernelPosI,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradPtsKernelPosJ,
        const torch::PackedTensorAccessor32<IndexT, 2, torch::RestrictPtrTraits> indexMap,
        bool grad,
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradOutMatrix,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradPtsKernelI,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradPtsKernelJ,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradIKernel,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradJKernel,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradGradPtsKernelPosI,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradGradPtsKernelPosJ) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t iti = (blockIdx.y * blockDim.y) + threadIdx.y;

    using NN = NNIterator<3, ScalarT>;
    if (pi >= ptsPos.size(0) || iti >= NN::total()) {
        return;
    }

    auto iAcc = gridI->getAccessor();
    const nanovdb::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPos[pi]);
    auto it = NN(roundVec(piLocal), iti);

    if (!iAcc.isActive(*it)) {
        return;
    }
    const int64_t offsetI = iAcc.getValue(*it) - 1;

    // Evaluate kernel K(k, i)
    ScalarT kiF = 0.0, kiBk, kiDk;
    nanovdb::Vec3<ScalarT> gradKiF(0.0), kiDb(0.0);
    kernel_grad_evaluation_fwd(
            offsetI, pi, transformI.scale<ScalarT>(),
            piLocal[0] - (ScalarT) (*it)[0],
            piLocal[1] - (ScalarT) (*it)[1],
            piLocal[2] - (ScalarT) (*it)[2],
            ptsKernelI, iKernel, gradPtsKernelPosI,
            grad, kiF, gradKiF, kiBk, kiDk, kiDb);

    auto jAcc = gridJ->getAccessor();
    const nanovdb::Vec3<ScalarT> pjLocal = transformJ.apply<ScalarT>(ptsPos[pi]);

    // Iterate over index [j] (to be put into one kernel execution)
#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(roundVec(pjLocal)); jt.isValid(); ++jt) {
        if (!jAcc.isActive(*jt)) {
            continue;
        }
        const int64_t offsetJ = jAcc.getValue(*jt) - 1;

        // Evaluate kernel K(k, j)
        ScalarT kjF = 0.0, kjBk, kjDk;
        nanovdb::Vec3<ScalarT> gradKjF(0.0), kjDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, pi, transformJ.scale<ScalarT>(),
                pjLocal[0] - (ScalarT) (*jt)[0],
                pjLocal[1] - (ScalarT) (*jt)[1],
                pjLocal[2] - (ScalarT) (*jt)[2],
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                grad, kjF, gradKjF, kjBk, kjDk, kjDb);

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(it->asVec3s())));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        auto dummyAcc2 = ptsPos;
        auto dummyAcc1 = gradOutMatrix;

        // Backward (from I)
        kernel_grad_evaluation_bwd<ScalarT, true>(
                offsetI, pi,
                ptsKernelI, iKernel, gradPtsKernelPosI,
                gradOutMatrix, dummyAcc2, grad, 1.0,
                gradPtsKernelI, gradIKernel, dummyAcc1, gradGradPtsKernelPosI,
                outMatrixIdx, kjF, gradKjF,
                kiF, gradKiF, kiBk, kiDk, kiDb);

        // Backward (from J)
        kernel_grad_evaluation_bwd<ScalarT, true>(
                offsetJ, pi,
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                gradOutMatrix, dummyAcc2, grad, 1.0,
                gradPtsKernelJ, gradJKernel, dummyAcc1, gradGradPtsKernelPosJ,
                outMatrixIdx, kiF, gradKiF,
                kjF, gradKjF, kjBk, kjDk, kjDb);
    }
}

template <>
void dispatchMatrixBuildingBackward(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& gridI,
                                    const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& gridJ,
                                    const fvdb::VoxelCoordTransform& transformI,
                                    const fvdb::VoxelCoordTransform& transformJ,
                                    const torch::Tensor& ptsPos,
                                    const torch::Tensor& ptsKernelI,
                                    const torch::Tensor& ptsKernelJ,
                                    const torch::Tensor& iKernel,
                                    const torch::Tensor& jKernel,
                                    const torch::Tensor& gradPtsKernelPosI,
                                    const torch::Tensor& gradPtsKernelPosJ,
                                    const torch::Tensor& indexMap,
                                    bool grad,
                                    const torch::Tensor& gradOutMatrix,
                                    torch::Tensor& gradPtsKernelI,
                                    torch::Tensor& gradPtsKernelJ,
                                    torch::Tensor& gradIKernel,
                                    torch::Tensor& gradJKernel,
                                    torch::Tensor& gradGradPtsKernelPosI,
                                    torch::Tensor& gradGradPtsKernelPosJ,
                                    unsigned nThreadsX, unsigned nThreadsY) {

    const auto* gridGridI = gridI.deviceGrid<nanovdb::ValueIndex>();
    const auto* gridGridJ = gridJ.deviceGrid<nanovdb::ValueIndex>();
    if (!gridGridI || !gridGridJ) {
        throw std::runtime_error("Failed to get pointer for nanovdb index grid");
    }

    const int64_t PCOUNT = ptsPos.size(0);
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, float>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    if (indexMap.scalar_type() == torch::kInt32) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuildingBackward", [&]() {
            matrixBuildingBackward<scalar_t, int><<<nblocks, nthreads>>>(
                    gridGridI, gridGridJ, transformI, transformJ,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    grad,
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradPtsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradIKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradJKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradGradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else if (indexMap.scalar_type() == torch::kLong) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuildingBackward", [&]() {
            matrixBuildingBackward<scalar_t, int64_t><<<nblocks, nthreads>>>(
                    gridGridI, gridGridJ, transformI, transformJ,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    grad,
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradPtsKernelI.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernelJ.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradIKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradJKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGradPtsKernelPosI.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradGradPtsKernelPosJ.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Scalar type of indexer is not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
