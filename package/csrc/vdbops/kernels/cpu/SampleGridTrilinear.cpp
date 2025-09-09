#include "Kernels.h"
#include "utils/Utils.h"
#include "utils/TrilinearInterpolationWithGradIterator.h"


template <typename ScalarT>
void sampleGridTrilinear(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                         const torch::TensorAccessor<ScalarT, 2> points,
                         const torch::TensorAccessor<ScalarT, 2> gridData,
                         torch::TensorAccessor<ScalarT, 2> outFeatures,
                         fvdb::VoxelCoordTransform transform) {
    auto dualAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);

        for (auto it = fvdb::TrilinearInterpolationIterator<ScalarT>(xyz); it.isValid(); ++it) {
            const ScalarT wTrilinear = it->second;
            const nanovdb::Coord ijk = it->first;
            if (dualAcc.isActive(ijk)) {
                const int64_t indexIjk = voxelIndex(dualAcc, ijk);
                for (int j = 0; j < gridData.size(1); j += 1) {
                    outFeatures[pi][j] += wTrilinear * gridData[indexIjk][j];
                }
            }
        }
    }
}


template <typename ScalarT>
void sampleGridTrilinearWithGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                                 const torch::TensorAccessor<ScalarT, 2> points,
                                 const torch::TensorAccessor<ScalarT, 2> gridData,
                                 torch::TensorAccessor<ScalarT, 2> outFeatures,
                                 torch::TensorAccessor<ScalarT, 3> outGradFeatures,
                                 fvdb::VoxelCoordTransform transform) {
    auto dualAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);

        for (auto it = fvdb::TrilinearInterpolationWithGradIterator<ScalarT>(xyz); it.isValid(); ++it) {
            const nanovdb::Vec4<ScalarT> wXYZ = it->second;
            const nanovdb::Coord ijk = it->first;
            if (dualAcc.isActive(ijk)) {
                const int64_t indexIjk = voxelIndex(dualAcc, ijk);
                for (int j = 0; j < gridData.size(1); j += 1) {
                    outFeatures[pi][j] += wXYZ[0] * gridData[indexIjk][j];
                    auto scale = transform.template scale<ScalarT>();
                    for (int dim = 0; dim < 3; ++ dim) {
                        outGradFeatures[pi][j][dim] += wXYZ[dim + 1] * gridData[indexIjk][j] * scale;
                    }
                }
            }
        }
    }
}


template <typename ScalarT>
void sampleGridTrilinearWithGradGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* cpuGrid,
                                     const torch::TensorAccessor<ScalarT, 2> points,
                                     const torch::TensorAccessor<ScalarT, 2> gradOutFeatures,
                                     const torch::TensorAccessor<ScalarT, 3> gradOutGradFeatures,
                                     torch::TensorAccessor<ScalarT, 2> outGridData,
                                     fvdb::VoxelCoordTransform transform) {
    auto gridAcc = cpuGrid->getAccessor();

    for (int pi = 0; pi < points.size(0); pi += 1) {
        const nanovdb::Vec3<ScalarT> xyz = transform.apply<ScalarT>(points[pi]);

        for (auto it = fvdb::TrilinearInterpolationWithGradIterator<ScalarT>(xyz); it.isValid(); ++it) {
            const nanovdb::Vec4<ScalarT> wXYZ = it->second;
            const nanovdb::Coord ijk = it->first;
            if (gridAcc.isActive(ijk)) {
                const int64_t indexIjk = voxelIndex(gridAcc, ijk);
                for (int j = 0; j < gradOutFeatures.size(1); j += 1) {
                    outGridData[indexIjk][j] += wXYZ[0] * gradOutFeatures[pi][j];
                    auto scale = transform.template scale<ScalarT>();
                    for (int dim = 0; dim < 3; ++ dim) {
                        outGridData[indexIjk][j] += wXYZ[dim + 1] * gradOutGradFeatures[pi][j][dim] * scale;
                    }
                }
            }
        }
    }
}




namespace fvdb {

template <>
void dispatchSampleGridTrilinear<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                      const torch::Tensor& points,
                                                      const torch::Tensor& gridData,
                                                      const VoxelCoordTransform& transform,
                                                      torch::Tensor& outFeatures,
                                                      unsigned nThreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "sampleGridTrilinear", [&]() {
        sampleGridTrilinear<scalar_t>(
            grid, points.accessor<scalar_t, 2>(), gridData.accessor<scalar_t, 2>(),
            outFeatures.accessor<scalar_t, 2>(), transform);
    });
}


template <>
void dispatchSampleGridTrilinearWithGrad<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                              const torch::Tensor& points,
                                                              const torch::Tensor& gridData,
                                                              const VoxelCoordTransform& transform,
                                                              torch::Tensor& outFeatures,
                                                              torch::Tensor& outGradFeatures,
                                                              unsigned nThreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "sampleGridTrilinearWithGrad", [&]() {
        sampleGridTrilinearWithGrad<scalar_t>(
                grid, points.accessor<scalar_t, 2>(), gridData.accessor<scalar_t, 2>(),
                outFeatures.accessor<scalar_t, 2>(), outGradFeatures.accessor<scalar_t, 3>(),
                transform);
    });
}


template <>
void dispatchSampleGridTrilinearWithGradGrad<nanovdb::HostBuffer>(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridBuf,
                                                                  const torch::Tensor& points,
                                                                  const torch::Tensor& gradOutFeatures,
                                                                  const torch::Tensor& gradOutGradFeatures,
                                                                  const VoxelCoordTransform& transform,
                                                                  torch::Tensor& outGridData,
                                                                  unsigned nThreadsX, unsigned nThreadsY) {
    const auto* grid = gridBuf.template grid<nanovdb::ValueIndex>();
    if (!grid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "sampleGridTrilinearWithGradGrad", [&]() {
        sampleGridTrilinearWithGradGrad<scalar_t>(
                grid, points.accessor<scalar_t, 2>(), gradOutFeatures.accessor<scalar_t, 2>(),
                gradOutGradFeatures.accessor<scalar_t, 3>(), outGridData.accessor<scalar_t, 2>(), transform);
    });
}

} // namespace fvdb