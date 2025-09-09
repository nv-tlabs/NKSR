#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void ijkToIndex(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                const torch::TensorAccessor<ScalarT, 2> coords,
                torch::TensorAccessor<int64_t, 1> outIndex) {
    auto acc = cpuGrid->getAccessor();

    for (int ci = 0; ci < coords.size(0); ci += 1) {
        const nanovdb::Coord vox(coords[ci][0], coords[ci][1], coords[ci][2]);
        if (acc.isActive(vox)) {
            outIndex[ci] = voxelIndex(acc, vox);
        } else {
            outIndex[ci] = -1;
        }
    }
}


namespace fvdb {

template <>
void dispatchIjkToIndex<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                             const torch::Tensor& coords,
                                             torch::Tensor& outIndex,
                                             unsigned nThreadsX) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }
    AT_DISPATCH_INTEGRAL_TYPES(coords.scalar_type(), "ijkToIndex", [&]() {
        ijkToIndex<scalar_t>(grid,
                             coords.accessor<scalar_t, 2>(),
                             outIndex.accessor<int64_t, 1>());
    });
}

} // namespace fvdb