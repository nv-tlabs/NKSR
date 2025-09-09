#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void splatIntoGridTrilinearWithCounts(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                                      const torch::TensorAccessor<ScalarT, 2> pointsAcc,
                                      const torch::TensorAccessor<ScalarT, 2> pointsDataAcc,
                                      torch::TensorAccessor<ScalarT, 2> outPrimalAcc,
                                      torch::TensorAccessor<ScalarT, 1> outCountsAcc,
                                      const fvdb::VoxelCoordTransform& transform) {

    auto primalAcc = cpuGrid->getAccessor();
    for (int64_t pi = 0; pi < pointsAcc.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> p = transform.apply<ScalarT>(pointsAcc[pi]);
        for (auto it = fvdb::TrilinearInterpolationIterator<ScalarT>(p); it.isValid(); ++it) {
            if (primalAcc.isActive(it->first)) {
                const int64_t offset = voxelIndex(primalAcc, it->first);
                for (int64_t j = 0; j < pointsDataAcc.size(1); j += 1) {
                    outPrimalAcc[offset][j] += it->second * pointsDataAcc[pi][j];
                }
                outCountsAcc[offset] += static_cast<ScalarT>(1);
            }
        }
    }
}


template <typename ScalarT>
void splatIntoGridTrilinear(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                            const torch::TensorAccessor<ScalarT, 2> points,
                            const torch::TensorAccessor<ScalarT, 2> pointsData,
                            torch::TensorAccessor<ScalarT, 2> outGridData,
                            fvdb::VoxelCoordTransform transform) {
    auto gridAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);

        for (auto it = fvdb::TrilinearInterpolationIterator<ScalarT>(xyz); it.isValid(); ++it) {
            const ScalarT wTrilinear = it->second;
            const nanovdb::Coord ijk = it->first;
            if (gridAcc.isActive(ijk)) {
                const int64_t indexIjk = voxelIndex(gridAcc, ijk);
                for (int j = 0; j < pointsData.size(1); j += 1) {
                    outGridData[indexIjk][j] += wTrilinear * pointsData[pi][j];
                }
            }
        }
    }
}




namespace fvdb {

template <>
void dispatchSplatIntoGridTrilinear(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                    const torch::Tensor& points,
                                    const torch::Tensor& pointsData,
                                    const VoxelCoordTransform& transform,
                                    torch::Tensor& outGridData,
                                    unsigned nthreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "splatIntoGridTrilinear", [&]() {
        splatIntoGridTrilinear<scalar_t>(
            grid, points.accessor<scalar_t, 2>(), pointsData.accessor<scalar_t, 2>(),
            outGridData.accessor<scalar_t, 2>(), transform);
    });
}


template <>
void dispatchSplatIntoGridTrilinearWithCounts<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                                   const torch::Tensor& points,
                                                                   const torch::Tensor& pointPrimalData,
                                                                   const VoxelCoordTransform& transform,
                                                                   torch::Tensor& outPrimalData,
                                                                   torch::Tensor& outCounts,
                                                                   unsigned nThreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "splatIntoGridTrilinearWithCounts", [&]() {
        splatIntoGridTrilinearWithCounts<scalar_t>(
            grid, points.accessor<scalar_t, 2>(), pointPrimalData.accessor<scalar_t, 2>(),
            outPrimalData.accessor<scalar_t, 2>(), outCounts.accessor<scalar_t, 1>(),
            transform);
    });
}

} // namespace fvdb