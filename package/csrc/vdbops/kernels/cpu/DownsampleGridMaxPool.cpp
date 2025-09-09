#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void downsampleGridMaxPool(const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                           const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                           const torch::TensorAccessor<ScalarT, 2> fineData,
                           torch::TensorAccessor<ScalarT, 2> outCoarseData,
                           unsigned poolingFactor) {

    auto fineGridAcc = fineGrid->getAccessor();
    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(coarseGrid->tree()); it.isValid(); it++) {
        const nanovdb::Coord coarseIjk = it->first;
        const nanovdb::Coord fineIjk0(coarseIjk[0] * poolingFactor,
                                      coarseIjk[1] * poolingFactor,
                                      coarseIjk[2] * poolingFactor);

        for (int l = 0; l < outCoarseData.size(1); l += 1) {
            outCoarseData[it->second][l] = -std::numeric_limits<ScalarT>::infinity();
        }
        for (unsigned i = 0; i < poolingFactor; i += 1) {
            for (unsigned j = 0; j < poolingFactor; j += 1) {
                for (unsigned k = 0; k < poolingFactor; k += 1) {
                    nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                    int64_t fineIndex = voxelIndex(fineGridAcc, fineIjk);
                    if (fineIndex < 0) {
                        continue;
                    }
                    for (int l = 0; l < outCoarseData.size(1); l += 1) {
                        outCoarseData[it->second][l] = std::max(fineData[fineIndex][l], outCoarseData[it->second][l]);
                    }
                }
            }
        }
    }
}


template <typename ScalarT>
void downsampleGridMaxPoolGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                               const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                               const torch::TensorAccessor<ScalarT, 2> fineData,
                               const torch::TensorAccessor<ScalarT, 2> coarseGradOut,
                               torch::TensorAccessor<ScalarT, 2> outFineGradIn,
                               unsigned poolingFactor) {

    auto fineGridAcc = fineGrid->getAccessor();
    for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(coarseGrid->tree()); it.isValid(); it++) {
        const nanovdb::Coord coarseIjk = it->first;
        const nanovdb::Coord fineIjk0(coarseIjk[0] * poolingFactor,
                                      coarseIjk[1] * poolingFactor,
                                      coarseIjk[2] * poolingFactor);

        for (int l = 0; l < outFineGradIn.size(1); l += 1) {
            int64_t maxIndex = -1;
            ScalarT maxValue = -INFINITY;

            for (unsigned i = 0; i < poolingFactor; i += 1) {
                for (unsigned j = 0; j < poolingFactor; j += 1) {
                    for (unsigned k = 0; k < poolingFactor; k += 1) {
                        const nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                        const int64_t fineIndex = voxelIndex(fineGridAcc, fineIjk);
                        if (fineIndex < 0) {
                            continue;
                        }
                        const ScalarT fineValue = fineData[fineIndex][l];

                        if (fineValue > maxValue) {
                            maxIndex = fineIndex;
                            maxValue = fineValue;
                        }
                    }
                }
            }
            if (maxIndex >= 0) {
                outFineGradIn[maxIndex][l] = coarseGradOut[it->second][l];
            }
        }
    }
}




namespace fvdb {

template <>
void dispatchDownsampleGridMaxPool<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& fineGridHdl,
                                                        const nanovdb::GridHandle<nanovdb::HostBuffer>& coarseGridHdl,
                                                        torch::Tensor& fineData,
                                                        torch::Tensor& outCoarseData,
                                                        unsigned poolingFactor,
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

    AT_DISPATCH_FLOATING_TYPES(fineData.scalar_type(), "downsampleGridMaxPool", [&]() {
        downsampleGridMaxPool<scalar_t>(fineGrid, coarseGrid,
                                        fineData.accessor<scalar_t, 2>(),
                                        outCoarseData.accessor<scalar_t, 2>(),
                                        poolingFactor);
    });
}


template <>
void dispatchDownsampleGridMaxPoolGrad<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& coarseGridHdl,
                                                            const nanovdb::GridHandle<nanovdb::HostBuffer>& fineGridHdl,
                                                            const torch::Tensor& fineData,
                                                            const torch::Tensor& coarseGradOut,
                                                            torch::Tensor& outFineGradIn,
                                                            unsigned poolingFactor,
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

    AT_DISPATCH_FLOATING_TYPES(coarseGradOut.scalar_type(), "downsampleGridMaxPoolGrad", [&]() {
        downsampleGridMaxPoolGrad<scalar_t>(coarseGrid, fineGrid,
                                            fineData.accessor<scalar_t, 2>(),
                                            coarseGradOut.accessor<scalar_t, 2>(),
                                            outFineGradIn.accessor<scalar_t, 2>(),
                                            poolingFactor);
    });
}

} // namespace fvdb