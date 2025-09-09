#include "meshing.h"
#include "mc_data.h"
#include "../common/iter_util.h"
#include <utils/cuda/Math.cuh>

// For active nodes iteration, need some subdivision to divide computation
constexpr int AC_NUM_BLOCK = fvdb::IndexTree::LeafNodeType::NUM_VALUES;
constexpr int AC_EACH_CHUNK = 2;
constexpr int AC_NUM_CHUNKS = AC_NUM_BLOCK / AC_EACH_CHUNK;

/**
 * Build Primal Cube Connection Graph (index into the Dual grid)
 */

__global__ void primalCubeGraphKernel(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* primal,
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* dual,
        torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> graph) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = primal->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = primal->tree().template getFirstNode<0>()[li];

    auto dualAcc = dual->getAccessor();

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (leaf.isActive(lo)) {
            nanovdb::Coord primalCoord = leaf.offsetToGlobalCoord(lo);
#pragma unroll
            for (int offset = 0; offset < 8; ++offset) {
                const auto& dualCoord = primalCoord + nanovdb::Coord(
                        cubeRelTable[offset][0], cubeRelTable[offset][1], cubeRelTable[offset][2]);
                graph[leaf.getValue(lo) - 1][offset] = dualAcc.getValue(dualCoord) - 1;
            }
        }
    }
}

template<>
torch::Tensor primalCubeGraph(const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer> &primal,
                              const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer> &dual,
                              unsigned nThreadsX) {
    int64_t nPrimal = primal.numVoxels();
    torch::Tensor graph = torch::zeros({nPrimal, 8}, torch::TensorOptions().dtype(torch::kLong).device(primal.device()));

    const auto* primalGrid = primal.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();
    const auto* dualGrid = dual.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();

    {
        const int64_t NTHREADSX = nThreadsX;
        const int64_t NBLOCKSX = fvdb::GET_BLOCKS(
                primal.nanovdbGrid().grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS, NTHREADSX);

        dim3 nblocks(NBLOCKSX, 1, 1);
        dim3 nthreads(NTHREADSX, 1, 1);

        primalCubeGraphKernel<<<nblocks, nthreads>>>(
                primalGrid, dualGrid,
                graph.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return graph;
}

/**
 * Build Multi-level Dual Connection Graph (index into the Primal grid)
 */

template <int Stride>
__global__ void dualCubeGraphLayerKernel(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* primal,
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* dual,
        int64_t baseIdx,
        torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> graph) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = primal->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = primal->tree().template getFirstNode<0>()[li];

    auto dualAcc = dual->getAccessor();

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;

        if (leaf.isActive(lo)) {
            nanovdb::Coord primalCoord = leaf.offsetToGlobalCoord(lo);
#pragma unroll
            for (auto cubeIt = CubeFaceIterator<Stride>(primalCoord); cubeIt.isValid(); cubeIt++) {
                if (!dualAcc.isActive(*cubeIt)) {
                    continue;
                }
                int64_t vi = dualAcc.getValue(*cubeIt) - 1;
                for (int ai = 0; ai < cubeIt.getAccCount(); ++ai) {
                    graph[vi][cubeIt.getAccInds(ai)] = leaf.getValue(lo) - 1 + baseIdx;
                }
            }
        }
    }
}

template<>
torch::Tensor dualCubeGraph(const std::vector<std::shared_ptr<SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>>>& primalGrids,
                            const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer> &corner,
                            unsigned nThreadsX) {
    int64_t currentBaseIdx = 0;

    int64_t nCorner = corner.numVoxels();
    torch::Tensor graph = torch::full({nCorner, 8}, -1, torch::TensorOptions().dtype(torch::kLong).device(corner.device()));
    auto graphAcc = graph.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>();
    const auto* dualGrid = corner.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();

    for (int l = 0; l < primalGrids.size(); ++l) {
        const auto primalSVH = primalGrids[l];
        if (primalSVH == nullptr)
            continue;
        if (primalSVH->numVoxels() <= 0)
            continue;

        const auto* primalGrid = primalSVH->nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();
        const int64_t NTHREADSX = nThreadsX;
        const int64_t NBLOCKSX = fvdb::GET_BLOCKS(
                primalSVH->nanovdbGrid().grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS, NTHREADSX);

        dim3 nblocks(NBLOCKSX, 1, 1);
        dim3 nthreads(NTHREADSX, 1, 1);

        switch (l) {
            case 0:
                dualCubeGraphLayerKernel<1><<<nblocks, nthreads>>>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 1:
                dualCubeGraphLayerKernel<2><<<nblocks, nthreads>>>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 2:
                dualCubeGraphLayerKernel<4><<<nblocks, nthreads>>>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 3:
                dualCubeGraphLayerKernel<8><<<nblocks, nthreads>>>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            default:
                throw std::runtime_error("DMC with large stride not instantiated!");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        currentBaseIdx += primalSVH->numVoxels();
    }

    torch::Tensor graphMask = torch::all(graph != -1, 1);
    return graph.index({graphMask});
}
