//
// Created by huangjh on 2022/12/5.
//

#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;


/**
 * Kernel Evaluation
 */

template <typename ScalarT>
void kernelEvaluation(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                      const fvdb::VoxelCoordTransform& transform,
                      const torch::TensorAccessor<ScalarT, 2> query,
                      const torch::TensorAccessor<ScalarT, 2> queryKernel,
                      const torch::TensorAccessor<ScalarT, 2> gridKernel,
                      const torch::TensorAccessor<ScalarT, 1> gridAlpha,
                      const torch::TensorAccessor<ScalarT, 3> gradKernelQuery,
                      torch::TensorAccessor<ScalarT, 1> outFunc,
                      torch::TensorAccessor<ScalarT, 2> outGradFunc) {

    const bool grad = outGradFunc.size(0) > 0;
    for (int64_t pi = 0; pi < query.size(0); pi += 1) {
        auto primalAcc = cpuGrid->getAccessor();
        const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(query[pi]);

        auto func = static_cast<ScalarT>(0.0);
        nanovdb::Vec3<ScalarT> dfunc(0.0);

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
                    queryKernel, gridKernel, gradKernelQuery,
                    grad, kiv, gradKiv, bk, dk, db);

            func += gridAlpha[offset] * kiv;
            dfunc += gridAlpha[offset] * gradKiv;
        }

        // Write result for this point.
        outFunc[pi] = func;
        if (grad) {
            for (int dim = 0; dim < 3; ++dim) {
                outGradFunc[pi][dim] = dfunc[dim];
            }
        }
    }
}

template <>
void dispatchKernelEvaluation(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                              const fvdb::VoxelCoordTransform& transform,
                              const torch::Tensor& query,
                              const torch::Tensor& queryKernel,
                              const torch::Tensor& gridKernel,
                              const torch::Tensor& gridAlpha,
                              const torch::Tensor& gradKernelQuery,
                              torch::Tensor& outFunc,
                              torch::Tensor& outGradFunc,
                              unsigned nThreadsX) {
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluation", [&]() {
        kernelEvaluation<scalar_t>(
            gridGrid, transform,
            query.accessor<scalar_t, 2>(),
            queryKernel.accessor<scalar_t, 2>(),
            gridKernel.accessor<scalar_t, 2>(),
            gridAlpha.accessor<scalar_t, 1>(),
            gradKernelQuery.accessor<scalar_t, 3>(),
            outFunc.accessor<scalar_t, 1>(),
            outGradFunc.accessor<scalar_t, 2>());
    });
}

/**
 * Kernel Evaluation (Backward)
 */

template <typename ScalarT>
void kernelEvaluationBackward(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                              const fvdb::VoxelCoordTransform& transform,
                              const torch::TensorAccessor<ScalarT, 2> query,
                              const torch::TensorAccessor<ScalarT, 2> queryKernel,
                              const torch::TensorAccessor<ScalarT, 2> gridKernel,
                              const torch::TensorAccessor<ScalarT, 1> gridAlpha,
                              const torch::TensorAccessor<ScalarT, 3> gradKernelQuery,

                              const torch::TensorAccessor<ScalarT, 1> gradOutFunc,
                              const torch::TensorAccessor<ScalarT, 2> gradOutGradFunc,

                              torch::TensorAccessor<ScalarT, 2> gradQueryKernel,
                              torch::TensorAccessor<ScalarT, 2> gradGridKernel,
                              torch::TensorAccessor<ScalarT, 1> gradGridAlpha,
                              torch::TensorAccessor<ScalarT, 3> gradGradKernelQuery) {

    const bool grad = gradOutGradFunc.size(0) > 0;
    for (int64_t pi = 0; pi < query.size(0); pi += 1) {
        auto primalAcc = cpuGrid->getAccessor();
        const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(query[pi]);

        // For each point, iterate through all its neighbours.
        for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
            if (!primalAcc.isActive(*it)) {
                continue;
            }
            const int64_t offset = voxelIndex(primalAcc, *it);
            ScalarT alpha = gridAlpha[offset];

            // Kernel (and gradient) evaluation
            ScalarT kiv = 0.0, bk, dk;
            nanovdb::Vec3<ScalarT> gradKiv(0.0), db(0.0);
            kernel_grad_evaluation_fwd(
                    offset, pi, transform.scale<ScalarT>(),
                    p[0] - (ScalarT) (*it)[0],
                    p[1] - (ScalarT) (*it)[1],
                    p[2] - (ScalarT) (*it)[2],
                    queryKernel, gridKernel, gradKernelQuery,
                    grad, kiv, gradKiv, bk, dk, db);

            // Backprop (through the function part)
            kernel_grad_evaluation_bwd<ScalarT, false>(
                    offset, pi,
                    queryKernel, gridKernel, gradKernelQuery,
                    gradOutFunc, gradOutGradFunc,
                    false, alpha,
                    gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery,
                    -1, (ScalarT) 0.0, nanovdb::Vec3<ScalarT>(0.0),
                    kiv, gradKiv, bk, dk, db);

            if (grad) {
                // Backprop (through the grad function part)
                kernel_grad_evaluation_bwd<ScalarT, false>(
                        offset, pi,
                        queryKernel, gridKernel, gradKernelQuery,
                        gradOutFunc, gradOutGradFunc,
                        true, alpha,
                        gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery,
                        -1, (ScalarT) 0.0, nanovdb::Vec3<ScalarT>(0.0),
                        kiv, gradKiv, bk, dk, db);
            }

        }
    }
}

template <>
void dispatchKernelEvaluationBackward(const nanovdb::GridHandle<nanovdb::HostBuffer>& grid,
                                      const fvdb::VoxelCoordTransform& transform,
                                      const torch::Tensor& query,
                                      const torch::Tensor& queryKernel,
                                      const torch::Tensor& gridKernel,
                                      const torch::Tensor& gridAlpha,
                                      const torch::Tensor& gradKernelQuery,
                                      const torch::Tensor& gradOutFunc,
                                      const torch::Tensor& gradOutGradFunc,
                                      torch::Tensor& gradQueryKernel,
                                      torch::Tensor& gradGridKernel,
                                      torch::Tensor& gradGridAlpha,
                                      torch::Tensor& gradGradKernelQuery,
                                      unsigned nThreadsX) {
    const auto* gridGrid = grid.grid<nanovdb::ValueIndex>();
    AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluationBackward", [&]() {
        kernelEvaluationBackward<scalar_t>(
                gridGrid, transform,
                query.accessor<scalar_t, 2>(),
                queryKernel.accessor<scalar_t, 2>(),
                gridKernel.accessor<scalar_t, 2>(),
                gridAlpha.accessor<scalar_t, 1>(),
                gradKernelQuery.accessor<scalar_t, 3>(),
                gradOutFunc.accessor<scalar_t, 1>(),
                gradOutGradFunc.accessor<scalar_t, 2>(),
                gradQueryKernel.accessor<scalar_t, 2>(),
                gradGridKernel.accessor<scalar_t, 2>(),
                gradGridAlpha.accessor<scalar_t, 1>(),
                gradGradKernelQuery.accessor<scalar_t, 3>());
    });
}

/**
 * Build COO Indexer
 */

template <>
torch::Tensor buildCOOIndexer(const SparseFeatureIndexGrid<nanovdb::HostBuffer>& iSVH,
                              const SparseFeatureIndexGrid<nanovdb::HostBuffer>& jSVH,
                              unsigned nThreadsX) {
    unsigned iSize = iSVH.numVoxels();
    torch::Tensor indexer = torch::full({iSize, 125}, -1, torch::TensorOptions().dtype(torch::kInt32).device(iSVH.device()));
    auto indexerAcc = indexer.accessor<int, 2>();

    const auto* iGrid = iSVH.nanovdbGrid().grid<nanovdb::ValueIndex>();
    const auto* jGrid = jSVH.nanovdbGrid().grid<nanovdb::ValueIndex>();

    using NNIt5 = NNIterator<5, float>;
    const auto& primalRange = nanovdb::Vec3<float>(2.5);

    auto jPrimalAcc = jGrid->getAccessor();
    for (auto it = fvdb::ActiveVoxelIterator<IndexTree, -1>(iGrid->tree()); it.isValid(); it++) {
        const auto& iPrimal = it->first.asVec3s();
        const auto& ijWorld = iSVH.primalTransform().applyInv(iPrimal);
        const auto& jcPrimal = jSVH.primalTransform().apply(ijWorld);
        for (auto jt = NNIt5(jcPrimal); jt.isValid(); ++jt) {
            if (!jPrimalAcc.isActive(*jt)) {
                continue;
            }
            const auto& jPrimal = jt->asVec3s();
            if (!has_overlap(
                    iSVH.primalTransform().applyInv(iPrimal - primalRange),
                    iSVH.primalTransform().applyInv(iPrimal + primalRange),
                    jSVH.primalTransform().applyInv(jPrimal - primalRange),
                    jSVH.primalTransform().applyInv(jPrimal + primalRange))) {
                continue;
            }
            indexerAcc[it->second][jt.getCount()] = voxelIndex(jPrimalAcc, *jt);
        }
    }

    return indexer;
}
