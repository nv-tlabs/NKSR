#include "Kernels.h"
#include "utils/Utils.h"
#include "utils/BezierInterpolationIterator.h"


template <typename ScalarT>
void splatIntoGridBezier(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                            const torch::TensorAccessor<ScalarT, 2> points,
                            const torch::TensorAccessor<ScalarT, 2> pointsData,
                            torch::TensorAccessor<ScalarT, 2> outGridData,
                            fvdb::VoxelCoordTransform transform) {
    auto gridAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);

        for (auto it = fvdb::BezierInterpolationIterator<ScalarT>(xyz); it.isValid(); ++it) {
            const ScalarT wBezier = it->second;
            const nanovdb::Coord ijk = it->first;
            if (gridAcc.isActive(ijk)) {
                const int64_t indexIjk = voxelIndex(gridAcc, ijk);
                for (int j = 0; j < pointsData.size(1); j += 1) {
                    outGridData[indexIjk][j] += wBezier * pointsData[pi][j];
                }
            }
        }
    }
}




namespace fvdb {

template <>
void dispatchSplatIntoGridBezier(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                    const torch::Tensor& points,
                                    const torch::Tensor& pointsData,
                                    const VoxelCoordTransform& transform,
                                    torch::Tensor& outGridData,
                                    unsigned nthreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "splatIntoGridBezier", [&]() {
        splatIntoGridBezier<scalar_t>(
            grid, points.accessor<scalar_t, 2>(), pointsData.accessor<scalar_t, 2>(),
            outGridData.accessor<scalar_t, 2>(), transform);
    });
}

} // namespace fvdb