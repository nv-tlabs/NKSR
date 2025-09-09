#include "keval.h"
#include "../common/iter_util.h"

/**
 * K Building
 */

template <typename ScalarT>
void kBuilding(const nanovdb::NanoGrid<nanovdb::ValueIndex>* grid,
               const fvdb::VoxelCoordTransform& transform,
               const torch::TensorAccessor<ScalarT, 2> kernel,
               const torch::TensorAccessor<int64_t, 2> indexMap,   // long Tensor (I, 125)
               torch::TensorAccessor<ScalarT, 1> outMatrix,
               torch::TensorAccessor<ScalarT, 3> dummy3) {

    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(grid->tree()); it.isValid(); it++) {
        auto acc = grid->getAccessor();

        const int64_t offsetI = it->second;
        for (auto jt = NNIterator<3, ScalarT>(it->first); jt.isValid(); ++jt) {
            if (!acc.isActive(*jt)) {
                continue;
            }
            nanovdb::Coord diffIJ = (*jt) - it->first;
            const int64_t offsetJ = voxelIndex(acc, *jt);

            // Evaluate kernel K(i, j)
            ScalarT ijF = 0.0, ijBk, ijDk;
            nanovdb::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
            kernel_grad_evaluation_fwd(
                    offsetJ, offsetI, transform.scale<ScalarT>(),
                    (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                    kernel, kernel, dummy3,
                    false, ijF, gradIjF, ijBk, ijDk, ijDb);

            int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
            int64_t outMatrixIdx = indexMap[offsetI][indexColIdx];

            outMatrix[outMatrixIdx] += ijF;
        }

    }
}

template <>
void dispatchKBuilding(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                       const fvdb::VoxelCoordTransform& transform,
                       const torch::Tensor& kernel,
                       const torch::Tensor& indexMap,
                       torch::Tensor& outMatrix,
                       unsigned nThreadsX) {
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());
    AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuilding", [&]() {
        kBuilding<scalar_t>(
                gridGrid, transform,
                kernel.accessor<scalar_t, 2>(),
                indexMap.accessor<int64_t , 2>(),
                outMatrix.accessor<scalar_t, 1>(),
                dummy3.accessor<scalar_t, 3>());
    });
}

template <typename ScalarT>
void kBuildingBackward(const nanovdb::NanoGrid<nanovdb::ValueIndex>* grid,
                       const fvdb::VoxelCoordTransform& transform,
                       const torch::TensorAccessor<ScalarT, 2> kernel,
                       const torch::TensorAccessor<int64_t, 2> indexMap,   // long Tensor (I, 125)
                       const torch::TensorAccessor<ScalarT, 1> gradOutMatrix,
                       torch::TensorAccessor<ScalarT, 2> gradKernel,
                       torch::TensorAccessor<ScalarT, 3> dummy3) {

    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(grid->tree()); it.isValid(); it++) {
        auto acc = grid->getAccessor();

        const int64_t offsetI = it->second;
        for (auto jt = NNIterator<3, ScalarT>(it->first); jt.isValid(); ++jt) {
            if (!acc.isActive(*jt)) {
                continue;
            }
            nanovdb::Coord diffIJ = (*jt) - it->first;
            const int64_t offsetJ = voxelIndex(acc, *jt);

            // Evaluate kernel K(i, j)
            ScalarT ijF = 0.0, ijBk, ijDk;
            nanovdb::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
            kernel_grad_evaluation_fwd(
                    offsetJ, offsetI, transform.scale<ScalarT>(),
                    (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                    kernel, kernel, dummy3,
                    false, ijF, gradIjF, ijBk, ijDk, ijDb);

            int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
            int64_t outMatrixIdx = indexMap[offsetI][indexColIdx];

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
void dispatchKBuildingBackward(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                               const fvdb::VoxelCoordTransform& transform,
                               const torch::Tensor& kernel,
                               const torch::Tensor& indexMap,
                               const torch::Tensor& gradOutMatrix,
                               torch::Tensor& gradKernel,
                               unsigned nThreadsX) {
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());
    AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "kBuildingBackward", [&]() {
        kBuildingBackward<scalar_t>(
                gridGrid, transform,
                kernel.accessor<scalar_t, 2>(),
                indexMap.accessor<int64_t , 2>(),
                gradOutMatrix.accessor<scalar_t, 1>(),
                gradKernel.accessor<scalar_t, 2>(),
                dummy3.accessor<scalar_t, 3>());
    });
}
