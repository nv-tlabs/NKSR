#include "meshing.h"
#include "mc_data.h"
#include "../common/iter_util.h"

/**
 * Build Primal Cube Connection Graph (index into the Dual grid)
 */
template<>
torch::Tensor primalCubeGraph(const SparseFeatureIndexGrid<nanovdb::HostBuffer> &primal,
                              const SparseFeatureIndexGrid<nanovdb::HostBuffer> &dual,
                              unsigned nThreadsX) {
    int64_t nPrimal = primal.numVoxels();
    torch::Tensor graph = torch::zeros({nPrimal, 8}, torch::TensorOptions().dtype(torch::kLong).device(primal.device()));
    auto graphAcc = graph.accessor<int64_t, 2>();

    const auto* dualGrid = dual.nanovdbGrid().grid<nanovdb::ValueIndex>();
    const auto* primalGrid = primal.nanovdbGrid().grid<nanovdb::ValueIndex>();

    auto dualAcc = dualGrid->getAccessor();
    for (auto pit = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(primalGrid->tree()); pit.isValid(); pit++) {
        const auto& primalCoord = pit->first;
        for (int offset = 0; offset < 8; ++offset) {
            const auto& dualCoord = primalCoord + nanovdb::Coord(
                    cubeRelTable[offset][0], cubeRelTable[offset][1], cubeRelTable[offset][2]);
            int64_t vi = voxelIndex(dualAcc, dualCoord);
            graphAcc[pit->second][offset] = vi;
        }
    }

    return graph;
}

/**
 * Build Multi-level Dual Connection Graph (index into the Primal grid)
 */

template <int Stride>
static void dualCubeGraphLayer(const nanovdb::NanoGrid<nanovdb::ValueIndex>* primalGrid,
                               const nanovdb::NanoGrid<nanovdb::ValueIndex>* dualGrid,
                               int64_t baseIdx,
                               torch::TensorAccessor<int64_t, 2> graph) {

    auto dualAcc = dualGrid->getAccessor();
    for (auto pit = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(primalGrid->tree()); pit.isValid(); pit++) {
        const auto& primalCoord = pit->first;
        for (auto cubeIt = CubeFaceIterator<Stride>(primalCoord); cubeIt.isValid(); cubeIt++) {
            if (!dualAcc.isActive(*cubeIt)) {
                continue;
            }
            int64_t vi = voxelIndex(dualAcc, *cubeIt);
            for (int ai = 0; ai < cubeIt.getAccCount(); ++ai) {
                graph[vi][cubeIt.getAccInds(ai)] = pit->second + baseIdx;
            }
        }
    }

}

template <>
torch::Tensor dualCubeGraph(const std::vector<std::shared_ptr<SparseFeatureIndexGrid<nanovdb::HostBuffer>>>& primalGrids,
                            const SparseFeatureIndexGrid<nanovdb::HostBuffer> &corner,
                            unsigned nThreadsX) {
    int64_t currentBaseIdx = 0;

    int64_t nCorner = corner.numVoxels();
    torch::Tensor graph = torch::full({nCorner, 8}, -1, torch::TensorOptions().dtype(torch::kLong).device(corner.device()));
    auto graphAcc = graph.accessor<int64_t, 2>();
    const auto* dualGrid = corner.nanovdbGrid().grid<nanovdb::ValueIndex>();

    for (size_t l = 0; l < primalGrids.size(); ++l) {
        const auto primalSVH = primalGrids[l];
        if (primalSVH == nullptr)
            continue;
        if (primalSVH->numVoxels() <= 0)
            continue;

        const auto* primalGrid = primalSVH->nanovdbGrid().grid<nanovdb::ValueIndex>();
        switch (l) {
            case 0:
                dualCubeGraphLayer<1>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 1:
                dualCubeGraphLayer<2>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 2:
                dualCubeGraphLayer<4>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            case 3:
                dualCubeGraphLayer<8>(primalGrid, dualGrid, currentBaseIdx, graphAcc); break;
            default:
                throw std::runtime_error("DMC with large stride not instantiated!");
        }
        currentBaseIdx += primalSVH->numVoxels();
    }

    torch::Tensor graphMask = torch::all(graph != -1, 1);
    return graph.index({graphMask});
}
