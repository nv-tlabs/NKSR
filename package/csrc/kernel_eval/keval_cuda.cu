#include "keval.h"
#include "../common/iter_util.h"

// Feature-VDB
#include <nanovdb/NanoVDB.h>
#include <utils/cuda/Math.cuh>
using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;


/**
 * Kernel Evaluation
 */

template <typename ScalarT>
__global__ void kernelEvaluation(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                 fvdb::VoxelCoordTransform transform,
                                 const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> query,
                                 const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> queryKernel,
                                 const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
                                 const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gridAlpha,
                                 const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelQuery,
                                 torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> outFunc,
                                 torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> outGradFunc) {
    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= query.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(query[pi]);
    const bool grad = outGradFunc.size(0) > 0;

    auto func = static_cast<ScalarT>(0.0);
    nanovdb::Vec3<ScalarT> dfunc(0.0);

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
                queryKernel, gridKernel, gradKernelQuery,
                grad, kiv, gradKiv, bk, dk, db);

        func += gridAlpha[offset] * kiv;
        dfunc += gridAlpha[offset] * gradKiv;
    }

    // Write result for this point.
    outFunc[pi] = func;
    if (grad) {
#pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            outGradFunc[pi][dim] = dfunc[dim];
        }
    }
}

template<>
void dispatchKernelEvaluation(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
                              const fvdb::VoxelCoordTransform& transform,
                              const torch::Tensor& query,
                              const torch::Tensor& queryKernel,
                              const torch::Tensor& gridKernel,
                              const torch::Tensor& gridAlpha,
                              const torch::Tensor& gradKernelQuery,
                              torch::Tensor& outFunc,
                              torch::Tensor& outGradFunc,
                              unsigned nThreadsX) {
    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = grid.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = query.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluation", [&]() {
        kernelEvaluation<scalar_t><<<nblocks, nthreads>>>(
            gpuGrid, transform,
            query.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            queryKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gridAlpha.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gradKernelQuery.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            outFunc.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            outGradFunc.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


/**
 * Kernel Evaluation (Backward)
 */

template <typename ScalarT>
__global__ void kernelEvaluationBackward(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                         fvdb::VoxelCoordTransform transform,
                                         const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> query,
                                         const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> queryKernel,
                                         const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gridKernel,
                                         const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gridAlpha,
                                         const torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradKernelQuery,
                                         const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradOutFunc,
                                         const torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradOutGradFunc,
                                         torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradQueryKernel,
                                         torch::PackedTensorAccessor32<ScalarT, 2, torch::RestrictPtrTraits> gradGridKernel,
                                         torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> gradGridAlpha,
                                         torch::PackedTensorAccessor32<ScalarT, 3, torch::RestrictPtrTraits> gradGradKernelQuery) {

    const int32_t pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pi >= query.size(0)) {
        return;
    }

    auto primalAcc = gpuGrid->getAccessor();
    const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(query[pi]);
    const bool grad = gradOutGradFunc.size(0) > 0;

    // For each point, iterate through all its neighbours.
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1;

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

template <>
void dispatchKernelEvaluationBackward(const nanovdb::GridHandle<fvdb::PytorchDeviceBuffer>& grid,
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

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = grid.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = query.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluationBackward", [&]() {
        kernelEvaluationBackward<scalar_t><<<nblocks, nthreads>>>(
                gpuGrid, transform,
                query.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                queryKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gridAlpha.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                gradKernelQuery.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                gradOutFunc.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                gradOutGradFunc.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradQueryKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradGridKernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradGridAlpha.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                gradGradKernelQuery.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/**
 * Build COO Indexer
 */

__global__ void cooIndexerKernel(const nanovdb::NanoGrid<nanovdb::ValueIndex>* iGrid,
                                 const nanovdb::NanoGrid<nanovdb::ValueIndex>* jGrid,
                                 fvdb::VoxelCoordTransform iTransform,
                                 fvdb::VoxelCoordTransform jTransform,
                                 torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> indexer) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = iGrid->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const IndexTree::LeafNodeType& leaf = iGrid->tree().template getFirstNode<0>()[li];
    using NNIt5 = NNIterator<5, float>;
    auto jPrimalAcc = jGrid->getAccessor();
    const auto& primalRange = nanovdb::Vec3<float>(2.5);

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {

        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (leaf.isActive(lo)) {
            nanovdb::Coord ijk = leaf.offsetToGlobalCoord(lo);

            const auto& iPrimal = ijk.asVec3s();
            const auto& ijWorld = iTransform.applyInv(iPrimal);
            const auto& jcPrimal = jTransform.apply(ijWorld);
            for (auto jt = NNIt5(jcPrimal); jt.isValid(); ++jt) {
                if (!jPrimalAcc.isActive(*jt)) {
                    continue;
                }
                const auto& jPrimal = jt->asVec3s();
                if (!has_overlap(
                        iTransform.applyInv(iPrimal - primalRange),
                        iTransform.applyInv(iPrimal + primalRange),
                        jTransform.applyInv(jPrimal - primalRange),
                        jTransform.applyInv(jPrimal + primalRange))) {
                    continue;
                }
                indexer[leaf.getValue(lo) - 1][jt.getCount()] = jPrimalAcc.getValue(*jt) - 1;
            }
        }
    }
}

template <>
torch::Tensor buildCOOIndexer(const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>& iSVH,
                              const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>& jSVH,
                              unsigned nThreadsX) {
    unsigned iSize = iSVH.numVoxels();
    torch::Tensor indexer = torch::full(
            {iSize, 125}, -1, torch::TensorOptions().dtype(torch::kInt32).device(iSVH.device()));

    const auto* iGrid = iSVH.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();
    const auto* jGrid = jSVH.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();

    const int64_t PCOUNT = iSVH.nanovdbGrid().grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS;
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    cooIndexerKernel<<<nblocks, nthreads>>>(
            iGrid, jGrid, iSVH.primalTransform(), jSVH.primalTransform(),
            indexer.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return indexer;
}

