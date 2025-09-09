#include "keval.h"
#include "../common/iter_util.h"

/**
 * Matrix Building
 */

template <typename ScalarT>
void matrixBuilding(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridI,
                    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridJ,
                    const fvdb::VoxelCoordTransform& transformI,
                    const fvdb::VoxelCoordTransform& transformJ,
                    const torch::TensorAccessor<ScalarT, 2> ptsPos,
                    const torch::TensorAccessor<ScalarT, 2> ptsKernelI,
                    const torch::TensorAccessor<ScalarT, 2> ptsKernelJ,
                    const torch::TensorAccessor<ScalarT, 2> iKernel,
                    const torch::TensorAccessor<ScalarT, 2> jKernel,
                    const torch::TensorAccessor<ScalarT, 3> gradPtsKernelPosI,
                    const torch::TensorAccessor<ScalarT, 3> gradPtsKernelPosJ,
                    const torch::TensorAccessor<int64_t, 2> indexMap,   // long Tensor (I, 125)
                    bool grad,          // Build GTG or QTQ
                    torch::TensorAccessor<ScalarT, 1> outMatrix) {

    // For each point (ind = k)
    for (int64_t pi = 0; pi < ptsPos.size(0); pi += 1) {

        auto iAcc = gridI->getAccessor();
        const nanovdb::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPos[pi]);

        // Iterate over index [i]
        for (auto it = NNIterator<3, ScalarT>(piLocal); it.isValid(); ++it) {
            if (!iAcc.isActive(*it)) {
                continue;
            }
            const int64_t offsetI = voxelIndex(iAcc, *it);

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
            for (auto jt = NNIterator<3, ScalarT>(pjLocal); jt.isValid(); ++jt) {
                if (!jAcc.isActive(*jt)) {
                    continue;
                }
                const int64_t offsetJ = voxelIndex(jAcc, *jt);

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

                nanovdb::Coord iC = transformJ.apply(transformI.applyInv(it->asVec3s())).round();
                int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
                int64_t outMatrixIdx = indexMap[offsetI][indexColIdx];

                outMatrix[outMatrixIdx] += outVal;
            }

        }
    }
}

template <>
void dispatchMatrixBuilding(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridI,
                            const nanovdb::GridHandle<nanovdb::HostBuffer>& gridJ,
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
    const auto* gridGridI = gridI.grid<nanovdb::ValueIndex>();
    const auto* gridGridJ = gridJ.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuilding", [&]() {
        matrixBuilding<scalar_t>(
                gridGridI, gridGridJ, transformI, transformJ,
                ptsPos.accessor<scalar_t, 2>(),
                ptsKernelI.accessor<scalar_t, 2>(),
                ptsKernelJ.accessor<scalar_t, 2>(),
                iKernel.accessor<scalar_t, 2>(),
                jKernel.accessor<scalar_t, 2>(),
                gradPtsKernelPosI.accessor<scalar_t, 3>(),
                gradPtsKernelPosJ.accessor<scalar_t, 3>(),
                indexMap.accessor<int64_t, 2>(),
                grad,
                outMatrix.accessor<scalar_t, 1>());
    });
}

template <typename ScalarT>
void matrixBuildingBackward(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridI,
                            const nanovdb::NanoGrid<nanovdb::ValueIndex>* gridJ,
                            const fvdb::VoxelCoordTransform& transformI,
                            const fvdb::VoxelCoordTransform& transformJ,
                            const torch::TensorAccessor<ScalarT, 2> ptsPos,
                            const torch::TensorAccessor<ScalarT, 2> ptsKernelI,
                            const torch::TensorAccessor<ScalarT, 2> ptsKernelJ,
                            const torch::TensorAccessor<ScalarT, 2> iKernel,
                            const torch::TensorAccessor<ScalarT, 2> jKernel,
                            const torch::TensorAccessor<ScalarT, 3> gradPtsKernelPosI,
                            const torch::TensorAccessor<ScalarT, 3> gradPtsKernelPosJ,
                            const torch::TensorAccessor<int64_t, 2> indexMap,
                            bool grad,
                            const torch::TensorAccessor<ScalarT, 1> gradOutMatrix,

                            torch::TensorAccessor<ScalarT, 2> gradPtsKernelI,
                            torch::TensorAccessor<ScalarT, 2> gradPtsKernelJ,
                            torch::TensorAccessor<ScalarT, 2> gradIKernel,
                            torch::TensorAccessor<ScalarT, 2> gradJKernel,
                            torch::TensorAccessor<ScalarT, 3> gradGradPtsKernelPosI,
                            torch::TensorAccessor<ScalarT, 3> gradGradPtsKernelPosJ) {

    // For each point (ind = k)
    for (int64_t pi = 0; pi < ptsPos.size(0); pi += 1) {

        auto iAcc = gridI->getAccessor();
        const nanovdb::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPos[pi]);

        // Iterate over index [i]
        for (auto it = NNIterator<3, ScalarT>(piLocal); it.isValid(); ++it) {
            if (!iAcc.isActive(*it)) {
                continue;
            }
            const int64_t offsetI = voxelIndex(iAcc, *it);

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
            for (auto jt = NNIterator<3, ScalarT>(pjLocal); jt.isValid(); ++jt) {
                if (!jAcc.isActive(*jt)) {
                    continue;
                }
                const int64_t offsetJ = voxelIndex(jAcc, *jt);

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

                nanovdb::Coord iC = transformJ.apply(transformI.applyInv(it->asVec3s())).round();
                int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
                int64_t outMatrixIdx = indexMap[offsetI][indexColIdx];

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
    }
}

template <>
void dispatchMatrixBuildingBackward(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridI,
                                    const nanovdb::GridHandle<nanovdb::HostBuffer>& gridJ,
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

    const auto* gridGridI = gridI.grid<nanovdb::ValueIndex>();
    const auto* gridGridJ = gridJ.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(ptsKernelI.scalar_type(), "matrixBuildingBackward", [&]() {
        matrixBuildingBackward<scalar_t>(
                gridGridI, gridGridJ, transformI, transformJ,
                ptsPos.accessor<scalar_t, 2>(),
                ptsKernelI.accessor<scalar_t, 2>(),
                ptsKernelJ.accessor<scalar_t, 2>(),
                iKernel.accessor<scalar_t, 2>(),
                jKernel.accessor<scalar_t, 2>(),
                gradPtsKernelPosI.accessor<scalar_t, 3>(),
                gradPtsKernelPosJ.accessor<scalar_t, 3>(),
                indexMap.accessor<int64_t, 2>(),
                grad,
                gradOutMatrix.accessor<scalar_t, 1>(),
                gradPtsKernelI.accessor<scalar_t, 2>(),
                gradPtsKernelJ.accessor<scalar_t, 2>(),
                gradIKernel.accessor<scalar_t, 2>(),
                gradJKernel.accessor<scalar_t, 2>(),
                gradGradPtsKernelPosI.accessor<scalar_t, 3>(),
                gradGradPtsKernelPosJ.accessor<scalar_t, 3>());
    });
}
