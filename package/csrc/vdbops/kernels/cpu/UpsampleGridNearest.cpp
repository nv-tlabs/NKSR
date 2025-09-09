#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void upsampleGridNearest(const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                         const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                         const torch::TensorAccessor<ScalarT, 2> coarseData,
                         torch::TensorAccessor<ScalarT, 2> outFineData,
                         unsigned upsamplingFactor) {

    auto coarseGridAcc = coarseGrid->getAccessor();
    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(fineGrid->tree()); it.isValid(); it++) {
        const nanovdb::Coord fineIjk = it->first;
        const nanovdb::Coord coarseIjk = nanovdb::Vec3<ScalarT>((ScalarT) fineIjk[0] / upsamplingFactor,
                                                                (ScalarT) fineIjk[1] / upsamplingFactor,
                                                                (ScalarT) fineIjk[2] / upsamplingFactor).floor();
        int64_t coarseIndex = voxelIndex(coarseGridAcc, coarseIjk);
        if (coarseIndex < 0) {
            continue;
        }
        for (int i = 0; i < outFineData.size(1); i += 1) {
            outFineData[it->second][i] = coarseData[coarseIndex][i];
        }
    }
}


template <typename ScalarT>
void upsampleGridNearestGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                             const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                             const torch::TensorAccessor<ScalarT, 2> fineData,
                             torch::TensorAccessor<ScalarT, 2> outCoarseData,
                             unsigned upsamplingFactor) {

    auto coarseGridAcc = coarseGrid->getAccessor();
    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(fineGrid->tree()); it.isValid(); it++) {
        const nanovdb::Coord fineIjk = it->first;
        const nanovdb::Coord coarseIjk = nanovdb::Vec3<ScalarT>((ScalarT) fineIjk[0] / upsamplingFactor,
                                                                (ScalarT) fineIjk[1] / upsamplingFactor,
                                                                (ScalarT) fineIjk[2] / upsamplingFactor).floor();
        int64_t coarseIndex = voxelIndex(coarseGridAcc, coarseIjk);
        if (coarseIndex < 0) {
            continue;
        }
        for (int i = 0; i < outCoarseData.size(1); i += 1) {
            outCoarseData[coarseIndex][i] += fineData[it->second][i];
        }
    }
}




namespace fvdb {

template <>
void dispatchUpsampleGridNearest<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& coarseGridHdl,
                                                      const nanovdb::GridHandle<nanovdb::HostBuffer>& fineGridHdl,
                                                      torch::Tensor& coarseData,
                                                      torch::Tensor& outFineData,
                                                      unsigned upsamplingFactor,
                                                      unsigned nThreadsX,
                                                      unsigned nThreadsY) {
    const auto* coarseGrid = coarseGridHdl.template grid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* fineGrid = fineGridHdl.template grid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }
    AT_DISPATCH_FLOATING_TYPES(coarseData.scalar_type(), "upsampleGridNearest", [&]() {
        upsampleGridNearest<scalar_t>(coarseGrid, fineGrid,
                                      coarseData.accessor<scalar_t, 2>(),
                                      outFineData.accessor<scalar_t, 2>(),
                                      upsamplingFactor);
    });

}


template <>
void dispatchUpsampleGridNearestGrad<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& fineGridHdl,
                                                          const nanovdb::GridHandle<nanovdb::HostBuffer>& coarseGridHdl,
                                                          torch::Tensor& fineData,
                                                          torch::Tensor& outCoarseData,
                                                          unsigned upsamplingFactor,
                                                          unsigned nThreadsX,
                                                          unsigned nThreadsY) {
    const auto* fineGrid = fineGridHdl.template grid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* coarseGrid = coarseGridHdl.template grid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(fineData.scalar_type(), "upsampleGridNearestGrad", [&]() {
        upsampleGridNearestGrad<scalar_t>(fineGrid, coarseGrid,
                                          fineData.accessor<scalar_t, 2>(),
                                          outCoarseData.accessor<scalar_t, 2>(),
                                          upsamplingFactor);
    });
}

} // namespace fvdb