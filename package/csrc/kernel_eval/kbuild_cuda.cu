#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
#include <utils/cuda/Math.cuh>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;

/**
 * K Building
 */

template <typename ScalarT, typename IndexT>
__global__ void kBuilding(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* grid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> kernel,
        const torch::PackedTensorAccessor32<IndexT, 2, torch::RestrictPtrTraits> indexMap,   // long Tensor (I, 125)
        torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> outMatrix,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> dummy3) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = grid->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const IndexTree::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[li];
    auto acc = grid->getAccessor();

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {

        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (!leaf.isActive(lo)) {
            continue;
        }

        const int64_t offsetI = leaf.getValue(lo) - 1;
        nanovdb::Coord ijk = leaf.offsetToGlobalCoord(lo);

#pragma unroll
        for (auto jt = NNIterator<3, ScalarT>(ijk); jt.isValid(); ++jt) {
            if (!acc.isActive(*jt)) {
                continue;
            }
            nanovdb::Coord diffIJ = (*jt) - ijk;
            const int64_t offsetJ = acc.getValue(*jt) - 1;

            // Evaluate kernel K(i, j)
            ScalarT ijF = 0.0, ijBk, ijDk;
            nanovdb::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
            kernel_grad_evaluation_fwd(
                    offsetJ, offsetI, transform.scale<ScalarT>(),
                    (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                    kernel, kernel, dummy3,
                    false, ijF, gradIjF, ijBk, ijDk, ijDb);

            int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
            IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

            outMatrix[outMatrixIdx] = ijF;
        }

    }

}

template <>
void dispatchKBuilding(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                       const fvdb::VoxelCoordTransform& transform,
                       const torch::Tensor& kernel,
                       const torch::Tensor& indexMap,
                       torch::Tensor& outMatrix,
                       unsigned nThreadsX) {
    const auto* gridGrid = grid.deviceGrid<nanovdb::ValueIndex>();
    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());

    const int64_t PCOUNT = grid.grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(nThreadsX, 1, 1);

    if (indexMap.scalar_type() == torch::kInt32) {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuilding", [&]() {
            kBuilding<scalar_t, int><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int , 2, torch::RestrictPtrTraits>(),
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    dummy3.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else if (indexMap.scalar_type() == torch::kLong) {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuilding", [&]() {
            kBuilding<scalar_t, int64_t><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int64_t , 2, torch::RestrictPtrTraits>(),
                    outMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    dummy3.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Scalar type of indexer is not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ScalarT, typename IndexT>
__global__ void kBuildingBackward(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* grid,
        fvdb::VoxelCoordTransform transform,
        const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> kernel,
        const torch::PackedTensorAccessor32<IndexT, 2, torch::RestrictPtrTraits> indexMap,   // long Tensor (I, 125)
        const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradOutMatrix,
        torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradKernel,
        torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> dummy3) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = grid->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const IndexTree::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[li];
    auto acc = grid->getAccessor();

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {

        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (!leaf.isActive(lo)) {
            continue;
        }

        const int64_t offsetI = leaf.getValue(lo) - 1;
        nanovdb::Coord ijk = leaf.offsetToGlobalCoord(lo);

#pragma unroll
        for (auto jt = NNIterator<3, ScalarT>(ijk); jt.isValid(); ++jt) {
            if (!acc.isActive(*jt)) {
                continue;
            }
            nanovdb::Coord diffIJ = (*jt) - ijk;
            const int64_t offsetJ = acc.getValue(*jt) - 1;

            // Evaluate kernel K(i, j)
            ScalarT ijF = 0.0, ijBk, ijDk;
            nanovdb::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
            kernel_grad_evaluation_fwd(
                    offsetJ, offsetI, transform.scale<ScalarT>(),
                    (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                    kernel, kernel, dummy3,
                    false, ijF, gradIjF, ijBk, ijDk, ijDb);

            int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
            IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

            auto dummy2 = gradKernel;
            auto dummy1 = gradOutMatrix;
            kernel_grad_evaluation_bwd<ScalarT, true>(
                    offsetJ, offsetI,
                    kernel, kernel, dummy3 /* useless*/ ,
                    gradOutMatrix, dummy2 /* useless*/ , false, 1.0,
                    gradKernel, gradKernel, dummy1 /* useless*/ , dummy3 /* useless*/ ,
                    outMatrixIdx, 1.0, gradIjF /* useless*/ ,
                    ijF, gradIjF, ijBk, ijDk, ijDb);
        }
    }
}

template <>
void dispatchKBuildingBackward(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                               const fvdb::VoxelCoordTransform& transform,
                               const torch::Tensor& kernel,
                               const torch::Tensor& indexMap,
                               const torch::Tensor& gradOutMatrix,
                               torch::Tensor& gradKernel,
                               unsigned nThreadsX) {
    const auto* gridGrid = grid.deviceGrid<nanovdb::ValueIndex>();
    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());

    const int64_t PCOUNT = grid.grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(nThreadsX, 1, 1);

    if (indexMap.scalar_type() == torch::kInt32) {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuildingBackward", [&]() {
            kBuildingBackward<scalar_t, int><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dummy3.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else if (indexMap.scalar_type() == torch::kLong) {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuildingBackward", [&]() {
            kBuildingBackward<scalar_t, int64_t><<<nblocks, nthreads>>>(
                    gridGrid, transform,
                    kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    indexMap.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                    gradOutMatrix.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    gradKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dummy3.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        });
    } else {
        throw std::runtime_error("Scalar type of indexer is not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
