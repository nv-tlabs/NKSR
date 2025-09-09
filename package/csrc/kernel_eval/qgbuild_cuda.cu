#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
#include <utils/cuda/Math.cuh>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;


template <typename ScalarT>
__forceinline__ __device__ static void putValue(
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits>& out,
        int i, int j, ScalarT f, const nanovdb::Vec3<ScalarT>& df) {
    out[i][j] = f;
}

template <typename ScalarT>
__forceinline__ __device__ static void putValue(
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits>& out,
        int i, int j, ScalarT f, const nanovdb::Vec3<ScalarT>& df) {
    out[i][j][0] = df[0];
    out[i][j][1] = df[1];
    out[i][j][2] = df[2];
}

template <typename ScalarT>
__forceinline__ __device__ static void getValue(
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits>& in,
        int i, int j, ScalarT& f, nanovdb::Vec3<ScalarT>& df) {
    f = in[i][j];
}

template <typename ScalarT>
__forceinline__ __device__ static void getValue(
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits>& in,
        int i, int j, ScalarT& f, nanovdb::Vec3<ScalarT>& df) {
    df[0] = in[i][j][0];
    df[1] = in[i][j][1];
    df[2] = in[i][j][2];
}

// Dispatch multiplication
template <typename ScalarT>
__inline__ __device__ static ScalarT valMult(
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> aVal,
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> bVal,
        int a, int b) {
    return aVal[a] * bVal[b];
}

template <typename ScalarT>
__inline__ __device__ static float valMult(
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> aVal,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> bVal,
        int a, int b) {
    return aVal[a][0] * bVal[b][0] + aVal[a][1] * bVal[b][1] + aVal[a][2] * bVal[b][2];
}

template <typename ScalarT>
__inline__ __device__ static void valMultBwd(
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> aVal,
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> bVal,
        int a, int b,
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradAVal,
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradBVal,
        ScalarT grad_fg) {
    gpuAtomicAddNoReturn(&gradAVal[a], grad_fg * bVal[b]);
    gpuAtomicAddNoReturn(&gradBVal[b], grad_fg * aVal[a]);
}

template <typename ScalarT>
__inline__ __device__ static void valMultBwd(
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> aVal,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> bVal,
        int a, int b,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradAVal,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradBVal,
        ScalarT grad_fg) {
    gpuAtomicAddNoReturn(&gradAVal[a][0], grad_fg * bVal[b][0]);
    gpuAtomicAddNoReturn(&gradAVal[a][1], grad_fg * bVal[b][1]);
    gpuAtomicAddNoReturn(&gradAVal[a][2], grad_fg * bVal[b][2]);
    gpuAtomicAddNoReturn(&gradBVal[b][0], grad_fg * aVal[a][0]);
    gpuAtomicAddNoReturn(&gradBVal[b][1], grad_fg * aVal[a][1]);
    gpuAtomicAddNoReturn(&gradBVal[b][2], grad_fg * aVal[a][2]);
}

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
 * Build Q & G CSR Matrices
 */

template <typename ScalarT, int Dim>
__global__ void qgBuilding(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> pts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelPts,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> indexer,
        torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> outQg) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= pts.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi][0], pts[pi][1], pts[pi][2]);

    // For each point, iterate through all its neighbours.
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(roundVec(p)); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1;

        // Kernel gradient evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, pi, transform.scale<ScalarT>(),
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                Dim == 3, kiv, gradKiv, bk, dk, db);

        indexer[pi][it.getCount()] = offset;
        putValue(outQg, pi, it.getCount(), kiv, gradKiv);
    }
}

template <typename ScalarT, int Dim>
__global__ void qgBuildingBackward(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> pts,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> ptsKernel,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
        const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelPts,
        const torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> gradOutQg,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradPtsKernel,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradGridKernel,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradGradKernelPts) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= pts.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi][0], pts[pi][1], pts[pi][2]);

    // For each point, iterate through all its neighbours.
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(roundVec(p)); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1;

        // Kernel gradient evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, pi, transform.scale<ScalarT>(),
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                Dim == 3, kiv, gradKiv, bk, dk, db);

        // Backward
        ScalarT gData;
        nanovdb::Vec3<ScalarT> qData;
        getValue(gradOutQg, pi, it.getCount(), gData, qData);

        auto dummyAcc2 = gradPtsKernel;
        auto dummyAcc1 = gradPtsKernel[0];
        kernel_grad_evaluation_bwd<ScalarT, true>(
                offset, pi,
                ptsKernel, gridKernel, gradKernelPts,
                dummyAcc1, dummyAcc2, Dim == 3, 1.0,
                gradPtsKernel, gradGridKernel, dummyAcc1, gradGradKernelPts,
                -1, gData, qData,
                kiv, gradKiv, bk, dk, db);
    }
}

/**
 * CSR Matrix product
 */

template <typename ScalarT, int Dim>
__global__ void csrMatrixMultiplication(
        fvdb::VoxelCoordTransform transformI,
        fvdb::VoxelCoordTransform transformJ,
        const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> coordsI,
        const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> coordsJ,
        const torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> iValue,
        const torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> jValue,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> iRowPtr,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> jRowPtr,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> iColInds,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> jColInds,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> indexMap,
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> outMatrix) {

    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= iRowPtr.size(0) - 1) {
        return;
    }

    int a = (blockIdx.y * blockDim.y) + threadIdx.y + iRowPtr[batchIdx];
    if (a >= iRowPtr[batchIdx + 1]) return;

    for (int b = jRowPtr[batchIdx]; b < jRowPtr[batchIdx + 1]; ++b) {
        float fg = valMult(iValue, jValue, a, b);

        int offsetI = iColInds[a];
        int offsetJ = jColInds[b];

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(
                nanovdb::Vec3<ScalarT>(
                        coordsI[offsetI][0], coordsI[offsetI][1], coordsI[offsetI][2]))));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(nanovdb::Coord(
                coordsJ[offsetJ][0] - iC[0],
                coordsJ[offsetJ][1] - iC[1],
                coordsJ[offsetJ][2] - iC[2]));
        int outMatrixIdx = indexMap[offsetI][indexColIdx];

        gpuAtomicAddNoReturn(&outMatrix[outMatrixIdx], fg);
    }
}

template <typename ScalarT, int Dim>
__global__ void csrMatrixMultiplicationBackward(
        fvdb::VoxelCoordTransform transformI,
        fvdb::VoxelCoordTransform transformJ,
        const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> coordsI,
        const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> coordsJ,
        const torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> iValue,
        const torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> jValue,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> iRowPtr,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> jRowPtr,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> iColInds,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> jColInds,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> indexMap,
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradOutMatrix,
        torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> gradIValue,
        torch::PackedTensorAccessor32<ScalarT, Dim, torch::RestrictPtrTraits> gradJValue) {

    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= iRowPtr.size(0) - 1) {
        return;
    }

    int a = (blockIdx.y * blockDim.y) + threadIdx.y + iRowPtr[batchIdx];
    if (a >= iRowPtr[batchIdx + 1]) return;

    for (int b = jRowPtr[batchIdx]; b < jRowPtr[batchIdx + 1]; ++b) {
        int offsetI = iColInds[a];
        int offsetJ = jColInds[b];

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(
                nanovdb::Vec3<ScalarT>(
                        coordsI[offsetI][0], coordsI[offsetI][1], coordsI[offsetI][2]))));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(nanovdb::Coord(
                coordsJ[offsetJ][0] - iC[0],
                coordsJ[offsetJ][1] - iC[1],
                coordsJ[offsetJ][2] - iC[2]));
        int outMatrixIdx = indexMap[offsetI][indexColIdx];

        float gradFg = gradOutMatrix[outMatrixIdx];
        valMultBwd<ScalarT>(iValue, jValue, a, b, gradIValue, gradJValue, gradFg);
    }
}

template <>
void dispatchQgBuilding(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                        const fvdb::VoxelCoordTransform& transform,
                        const torch::Tensor& ptsPos,
                        const torch::Tensor& ptsKernel,
                        const torch::Tensor& gridKernel,
                        const torch::Tensor& gradKernelPts,
                        torch::Tensor& outIndexer,
                        torch::Tensor& outMatrix,
                        unsigned nThreadsX) {
    const auto* gridGrid = grid.deviceGrid<nanovdb::ValueIndex>();
    if (!gridGrid) {
        throw std::runtime_error("Failed to get pointer for nanovdb index grid");
    }

    const int64_t PCOUNT = ptsPos.size(0);
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);

    dim3 nblocks(NBLOCKSX);
    dim3 nthreads(nThreadsX);

    if (outMatrix.ndimension() == 2) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgBuilding", [&]() {
            qgBuilding<scalar_t, 2><<<nblocks, nthreads>>>(
                gridGrid, transform,
                ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                outIndexer.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                outMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        });
    } else if (outMatrix.ndimension() == 3) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgBuilding", [&]() {
            qgBuilding<scalar_t, 3><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    outIndexer.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    outMatrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
void dispatchQgBuildingBackward(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
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
    const auto* gridGrid = grid.deviceGrid<nanovdb::ValueIndex>();
    if (!gridGrid) {
        throw std::runtime_error("Failed to get pointer for nanovdb index grid");
    }

    const int64_t PCOUNT = ptsPos.size(0);
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);

    dim3 nblocks(NBLOCKSX);
    dim3 nthreads(nThreadsX);

    if (gradOutMatrix.ndimension() == 2) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgBuilding", [&]() {
            qgBuildingBackward<scalar_t, 2><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradPtsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else if (gradOutMatrix.ndimension() == 3) {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgBuilding", [&]() {
            qgBuildingBackward<scalar_t, 3><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    ptsPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    ptsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPtsKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradGradKernelPts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void csrMatrixMultiplicationCUDA(
        const fvdb::VoxelCoordTransform& transformI,
        const fvdb::VoxelCoordTransform& transformJ,
        const torch::Tensor& coordsI, const torch::Tensor& coordsJ,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        torch::Tensor& outMatrix, unsigned nThreadsX, unsigned nThreadsY) {

    const int64_t PCOUNT = iRowPtr.size(0) - 1;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, float>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    if (iValue.ndimension() == 1) {
        AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
            csrMatrixMultiplication<scalar_t, 1><<<nblocks, nthreads>>>(
                    transformI, transformJ,
                    coordsI.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    coordsJ.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    iValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    jValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    iRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    iColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        });
    } else if (iValue.ndimension() == 2) {
        AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
            csrMatrixMultiplication<scalar_t, 2><<<nblocks, nthreads>>>(
                    transformI, transformJ,
                    coordsI.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    coordsJ.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    iValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    iColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

void csrMatrixMultiplicationBackwardCUDA(
        const fvdb::VoxelCoordTransform& transformI,
        const fvdb::VoxelCoordTransform& transformJ,
        const torch::Tensor& coordsI, const torch::Tensor& coordsJ,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        const torch::Tensor& gradOutMatrix,
        torch::Tensor& gradIValue,
        torch::Tensor& gradJValue,
        unsigned nThreadsX, unsigned nThreadsY) {

    const int64_t PCOUNT = iRowPtr.size(0) - 1;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, float>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    if (iValue.ndimension() == 1) {
        AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
            csrMatrixMultiplicationBackward<scalar_t, 1><<<nblocks, nthreads>>>(
                    transformI, transformJ,
                    coordsI.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    coordsJ.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    iValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    jValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    iRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    iColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradIValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradJValue.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        });
    } else if (iValue.ndimension() == 2) {
        AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
            csrMatrixMultiplicationBackward<scalar_t, 2><<<nblocks, nthreads>>>(
                    transformI, transformJ,
                    coordsI.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    coordsJ.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    iValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    jValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    iRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jRowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    iColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    jColInds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradIValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradJValue.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

}
