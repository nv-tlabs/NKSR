#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void pointsInGrid(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                         const torch::TensorAccessor<ScalarT, 2> points,
                         torch::TensorAccessor<bool, 1> outMask,
                         fvdb::VoxelCoordTransform transform) {
    auto primalAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);
        const nanovdb::Coord vox = xyz.round();
        if (primalAcc.isActive(vox)) {
            outMask[pi] = true;
        } else {
            outMask[pi] = false;
        }
    }
}




namespace fvdb {

template <>
void dispatchPointsInGrid<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                      const VoxelCoordTransform& transform,
                                                      const torch::Tensor& points, torch::Tensor& outMask,
                                                      unsigned nThreadsX) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "pointsInGrid", [&]() {
        pointsInGrid<scalar_t>(
            grid, points.accessor<scalar_t, 2>(), outMask.accessor<bool, 1>(), transform);
    });
}

} // namespace fvdb
