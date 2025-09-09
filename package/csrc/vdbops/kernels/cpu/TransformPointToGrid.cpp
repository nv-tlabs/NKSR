#include "Kernels.h"
#include "utils/Utils.h"


template <typename ScalarT>
void transformPointsToGrid(const torch::TensorAccessor<ScalarT, 2> ptCoordsAcc,
                           torch::TensorAccessor<ScalarT, 2> outCoordsAcc,
                           const fvdb::VoxelCoordTransform& tx) {

    for (int64_t i = 0; i < ptCoordsAcc.size(0); i += 1) {
        const nanovdb::Vec3<ScalarT> wci = tx.apply(ptCoordsAcc[i][0],
                                                    ptCoordsAcc[i][1],
                                                    ptCoordsAcc[i][2]);
        outCoordsAcc[i][0] = wci[0];
        outCoordsAcc[i][1] = wci[1];
        outCoordsAcc[i][2] = wci[2];
    }
}


template <typename ScalarT>
void transformPointsToGridGrad(const torch::TensorAccessor<ScalarT, 2> outGradAcc,
                               torch::TensorAccessor<ScalarT, 2> outGradInAcc,
                               const fvdb::VoxelCoordTransform& tx) {

    for (int64_t i = 0; i < outGradAcc.size(0); i += 1) {
        const nanovdb::Vec3<ScalarT> wci = tx.applyGrad(outGradAcc[i][0],
                                                        outGradAcc[i][1],
                                                        outGradAcc[i][2]);
        outGradInAcc[i][0] = wci[0] * outGradAcc[i][0];
        outGradInAcc[i][1] = wci[1] * outGradAcc[i][1];
        outGradInAcc[i][2] = wci[2] * outGradAcc[i][2];
    }
}


template <typename ScalarT>
void invTransformPointsToGrid(const torch::TensorAccessor<ScalarT, 2> ptCoordsAcc,
                              torch::TensorAccessor<ScalarT, 2> outCoordsAcc,
                              const fvdb::VoxelCoordTransform& tx) {

    for (int64_t i = 0; i < ptCoordsAcc.size(0); i += 1) {
        const nanovdb::Vec3<ScalarT> wci = tx.applyInv(ptCoordsAcc[i][0],
                                                       ptCoordsAcc[i][1],
                                                       ptCoordsAcc[i][2]);
        outCoordsAcc[i][0] = wci[0];
        outCoordsAcc[i][1] = wci[1];
        outCoordsAcc[i][2] = wci[2];
    }
}


template <typename ScalarT>
void invTransformPointsToGridGrad(const torch::TensorAccessor<ScalarT, 2> outGradAcc,
                                  torch::TensorAccessor<ScalarT, 2> outGradInAcc,
                                  const fvdb::VoxelCoordTransform& tx) {

    for (int64_t i = 0; i < outGradAcc.size(0); i += 1) {
        const nanovdb::Vec3<ScalarT> wci = tx.applyInvGrad(outGradAcc[i][0],
                                                           outGradAcc[i][1],
                                                           outGradAcc[i][2]);
        outGradInAcc[i][0] = wci[0] * outGradAcc[i][0];
        outGradInAcc[i][1] = wci[1] * outGradAcc[i][1];
        outGradInAcc[i][2] = wci[2] * outGradAcc[i][2];
    }
}




namespace fvdb {

template <>
void dispatchTransformPointsToGrid<nanovdb::HostBuffer>(const torch::Tensor& points,
                                                        const VoxelCoordTransform& transform,
                                                        torch::Tensor& outCoords,
                                                        unsigned nThreadsX) {
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "transformPointsToGrid", [&]() {
        transformPointsToGrid<scalar_t>(points.accessor<scalar_t, 2>(),
                                  outCoords.accessor<scalar_t, 2>(),
                                  transform);
    });
}


template <>
void dispatchInvTransformPointsToGrid<nanovdb::HostBuffer>(const torch::Tensor& points,
                                                           const VoxelCoordTransform& transform,
                                                           torch::Tensor& outCoords,
                                                           unsigned nThreadsX) {

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "invTransformPointsToGrid", [&]() {
        invTransformPointsToGrid<scalar_t>(points.accessor<scalar_t, 2>(),
                                     outCoords.accessor<scalar_t, 2>(),
                                     transform);
    });
}


template <>
void dispatchTransformPointsToGridGrad<nanovdb::HostBuffer>(const torch::Tensor& gradOut,
                                                            const VoxelCoordTransform& transform,
                                                            torch::Tensor& outGradIn,
                                                            unsigned nThreadsX) {
    AT_DISPATCH_FLOATING_TYPES(gradOut.scalar_type(), "transformPointsToGridGrad", [&]() {
        transformPointsToGridGrad<scalar_t>(gradOut.accessor<scalar_t, 2>(),
                                            outGradIn.accessor<scalar_t, 2>(),
                                            transform);
    });
}


template <>
void dispatchInvTransformPointsToGridGrad<nanovdb::HostBuffer>(const torch::Tensor& gradOut,
                                                               const VoxelCoordTransform& transform,
                                                               torch::Tensor& outGradIn,
                                                               unsigned nThreadsX) {
    AT_DISPATCH_FLOATING_TYPES(gradOut.scalar_type(), "invTransformPointsToGridGrad", [&]() {
        invTransformPointsToGridGrad<scalar_t>(gradOut.accessor<scalar_t, 2>(),
                                               outGradIn.accessor<scalar_t, 2>(),
                                               transform);
    });
}

} // namespace fvdb