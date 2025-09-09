#include "Kernels.h"
#include "utils/Utils.h"

namespace fvdb {

template <>
void dispatchActiveGridCoords<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                   torch::Tensor& outGridCoords,
                                                   unsigned nThreadsX) {

    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    auto outCoordAcc = outGridCoords.accessor<int64_t, 2>();

    for (auto it = ActiveVoxelIterator<IndexTree, -1>(grid->tree()); it.isValid(); it++) {
        const nanovdb::Coord voxIjk = it->first;
        for (int i = 0; i < 3; i += 1) { outCoordAcc[it->second][i] = voxIjk[i]; }
    }
}

} // namespace fvdb


