#include "keval.h"
#include "../common/iter_util.h"

/**
 * RHS Evaluation (only with gradient)
 */

template <typename ScalarT>
void rhsEvaluation(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                   const fvdb::VoxelCoordTransform& transform,
                   const torch::TensorAccessor<ScalarT, 2> pts,
                   const torch::TensorAccessor<ScalarT, 2> ptsKernel,
                   const torch::TensorAccessor<ScalarT, 2> gridKernel,
                   const torch::TensorAccessor<ScalarT, 3> gradKernelPts,
                   const torch::TensorAccessor<ScalarT, 2> ptsData,
                   torch::TensorAccessor<ScalarT, 1> outRhs) {

    for (int64_t pi = 0; pi < pts.size(0); pi += 1) {
        auto primalAcc = cpuGrid->getAccessor();
        const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi]);
        nanovdb::Vec3<ScalarT> pData(ptsData[pi][0], ptsData[pi][1], ptsData[pi][2]);

        // For each point, iterate through all its neighbours.
        for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
            if (!primalAcc.isActive(*it)) {
                continue;
            }
            const int64_t offset = voxelIndex(primalAcc, *it);

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
            outRhs[offset] += res;
        }
    }
}

template <>
void dispatchRhsEvaluation(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                           const fvdb::VoxelCoordTransform& transform,
                           const torch::Tensor& pts,
                           const torch::Tensor& ptsKernel,
                           const torch::Tensor& gridKernel,
                           const torch::Tensor& gradKernelPts,
                           const torch::Tensor& ptsData,
                           torch::Tensor& outRhs,
                           unsigned nThreadsX) {
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "rhsEvaluation", [&]() {
        rhsEvaluation<scalar_t>(
                gridGrid, transform,
                pts.accessor<scalar_t, 2>(),
                ptsKernel.accessor<scalar_t, 2>(),
                gridKernel.accessor<scalar_t, 2>(),
                gradKernelPts.accessor<scalar_t, 3>(),
                ptsData.accessor<scalar_t, 2>(),
                outRhs.accessor<scalar_t, 1>());
    });
}

template <typename ScalarT>
void rhsEvaluationBackward(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                           const fvdb::VoxelCoordTransform& transform,
                           const torch::TensorAccessor<ScalarT, 2> pts,
                           const torch::TensorAccessor<ScalarT, 2> ptsKernel,
                           const torch::TensorAccessor<ScalarT, 2> gridKernel,
                           const torch::TensorAccessor<ScalarT, 3> gradKernelPts,
                           const torch::TensorAccessor<ScalarT, 2> ptsData,
                           const torch::TensorAccessor<ScalarT, 1> gradOutRhs,
                           torch::TensorAccessor<ScalarT, 2> gradPtsKernel,
                           torch::TensorAccessor<ScalarT, 2> gradGridKernel,
                           torch::TensorAccessor<ScalarT, 3> gradGradKernelPts,
                           torch::TensorAccessor<ScalarT, 2> gradPtsData) {

    for (int64_t pi = 0; pi < pts.size(0); pi += 1) {
        auto primalAcc = cpuGrid->getAccessor();
        const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pts[pi]);
        nanovdb::Vec3<ScalarT> pData(ptsData[pi][0], ptsData[pi][1], ptsData[pi][2]);

        // For each point, iterate through all its neighbours.
        for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
            if (!primalAcc.isActive(*it)) {
                continue;
            }
            const int64_t offset = voxelIndex(primalAcc, *it);

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
}

template <>
void dispatchRhsEvaluationBackward(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
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
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "rhsEvaluationBackward", [&]() {
        rhsEvaluationBackward<scalar_t>(
                gridGrid, transform,
                pts.accessor<scalar_t, 2>(),
                ptsKernel.accessor<scalar_t, 2>(),
                gridKernel.accessor<scalar_t, 2>(),
                gradKernelPts.accessor<scalar_t, 3>(),
                ptsData.accessor<scalar_t, 2>(),
                gradOutRhs.accessor<scalar_t, 1>(),
                gradPtsKernel.accessor<scalar_t, 2>(),
                gradGridKernel.accessor<scalar_t, 2>(),
                gradGradKernelPts.accessor<scalar_t, 3>(),
                gradPtsData.accessor<scalar_t, 2>());
    });
}
