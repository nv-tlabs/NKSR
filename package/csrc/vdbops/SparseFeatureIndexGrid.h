#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>

#include "autograd/Functions.h"
#include "Kernels.h"
#include "utils/Utils.h"


template <typename BufferType>
struct SparseFeatureIndexGrid {

    using BufferT = BufferType;

    explicit SparseFeatureIndexGrid(double voxSize, double voxOrigin, c10::DeviceIndex deviceIndex = -1) {
        setTransform(voxSize, voxOrigin);

        if (deviceIndex < 0) {
            mDeviceIndex = c10::cuda::current_device();
        } else {
            mDeviceIndex = deviceIndex;
        }
    }

    SparseFeatureIndexGrid(SparseFeatureIndexGrid&) = delete;

    inline const double voxelSize() const {
        return mVoxSize;
    }

    inline void setVoxelSize(double voxelSize) {
        setTransform(voxelSize, voxelOrigin());
    }

    inline const double voxelOrigin() const {
        return mPrimalTransform.applyInv<double>(0., 0., 0.)[0];
    }

    inline void setVoxelOrigin(double voxelOrigin) {
        setTransform(mVoxSize, voxelOrigin);
    }

    inline int64_t valueArraySize() const {
        if (!mGrid.template grid<nanovdb::ValueIndex>()) {
            return 0;
        }
        return mGrid.template grid<nanovdb::ValueIndex>(0)->valueCount();
    }

    inline int64_t numVoxels() const {
        if (!mGrid.template grid<nanovdb::ValueIndex>()) {
            return 0;
        }
        return mGrid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount();
    }

    inline const nanovdb::GridHandle<BufferType>& nanovdbGrid() const {
        return mGrid;
    }

    inline c10::Device device() const {
        return fvdb::bufferToTorchDevice<BufferType>(mDeviceIndex);
    }

    inline const fvdb::VoxelCoordTransform& primalTransform() const {
        return mPrimalTransform;
    }

    torch::Tensor gridToWorld(const torch::Tensor& voxCoords) const;

    torch::Tensor worldToGrid(const torch::Tensor& worldCoords) const;

    torch::Tensor activeGridCoords() const;

    std::vector<torch::Tensor> splatTrilinear(const torch::Tensor& points,
                                               const torch::Tensor& pointPrimalData,
                                               bool returnCounts = false) const;

    std::vector<torch::Tensor> sampleTrilinear(const torch::Tensor& points,
                                               const torch::Tensor& dualData,
                                               bool returnGrad = false) const;

    std::vector<torch::Tensor> sampleBezier(const torch::Tensor& points,
                                            const torch::Tensor& dualData,
                                            bool returnGrad = false) const;

    torch::Tensor pointsInGrid(const torch::Tensor& pts) const;

    torch::Tensor ijkToIndex(const torch::Tensor& ijk) const;

    std::shared_ptr<SparseFeatureIndexGrid<BufferType>> dualGrid() const {
        std::shared_ptr<SparseFeatureIndexGrid<BufferType>> ret =
            std::make_shared<SparseFeatureIndexGrid<BufferType>>(mVoxSize, voxelOrigin());
        ret->buildDualFromPrimalGrid(*this);
        return ret;
    }

    std::shared_ptr<SparseFeatureIndexGrid<BufferType>> coarsenedGrid(unsigned branchFactor) const {
        std::shared_ptr<SparseFeatureIndexGrid<BufferType>> ret =
            std::make_shared<SparseFeatureIndexGrid<BufferType>>(mVoxSize, voxelOrigin());
            ret->buildCoarseFromFineGrid(*this, branchFactor);
            return ret;
    }

    std::shared_ptr<SparseFeatureIndexGrid<BufferType>> subdividedGrid(unsigned subdivFactor, const torch::Tensor& mask) const {
        std::shared_ptr<SparseFeatureIndexGrid<BufferType>> ret =
            std::make_shared<SparseFeatureIndexGrid<BufferType>>(mVoxSize, voxelOrigin());
            ret->buildFineFromCoarseGrid(*this, mask, subdivFactor);
            return ret;
    }

    /*
     * The following methods build out the topology of the index grids from a variety of sources
     * For example, by quantizing points and adding padding or by coarsening a grid.
     */
    void buildFromPointsWithPadding(const torch::Tensor& points,
                                    const nanovdb::Coord& padMin,
                                    const nanovdb::Coord& padMax);

    void buildFromPointsNearestNeighbors(const torch::Tensor& points);

    void buildFromVoxelCoordinates(const torch::Tensor& coords,
                                   const nanovdb::Coord& padMin,
                                   const nanovdb::Coord& padMax);

    void buildFromNanoVDBGridHandle(nanovdb::GridHandle<BufferType>& primalGrid) {
        mGrid = std::move(primalGrid);
        fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
    }

    template <typename FromBufferType>
    void moveFrom(SparseFeatureIndexGrid<FromBufferType>& grid) {
        this->mGrid = fvdb::move<FromBufferType, BufferType>(grid.mGrid);
        this->mPrimalTransform = grid.mPrimalTransform;
        this->mDualTransform = grid.mDualTransform;
        this->mVoxSize = grid.mVoxSize;
        fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
    }

    friend class SparseFeatureIndexGrid<nanovdb::HostBuffer>;
    friend class SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>;

private:
    void buildCoarseFromFineGrid(const SparseFeatureIndexGrid& fine_grid,
                                 unsigned branchFactor = 2);

    void buildFineFromCoarseGrid(const SparseFeatureIndexGrid& coarseGrid,
                                 const torch::Tensor& subdivMask,
                                 unsigned subdivFactor = 2);

    void buildDualFromPrimalGrid(const SparseFeatureIndexGrid& primalGrid) {
        mGrid = fvdb::buildPaddedGridFromGrid<BufferType, 0, 1>(primalGrid.mGrid);
        mPrimalTransform = primalGrid.mDualTransform;
        mDualTransform = primalGrid.mPrimalTransform;
        fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
    }

    nanovdb::GridHandle<BufferType> mGrid;

    fvdb::VoxelCoordTransform mPrimalTransform;
    fvdb::VoxelCoordTransform mDualTransform;

    double mVoxSize = -1.0;

    c10::DeviceIndex mDeviceIndex = fvdb::defaultDeviceIndex<BufferType>();

    inline std::pair<double, double> coarseVoxSizeAndOrigin(unsigned branchingFactor = 2) const {
        const double w = static_cast<double>(branchingFactor) * mVoxSize;
        const double tx =  static_cast<double>(branchingFactor - 1) * mVoxSize * 0.5 + voxelOrigin();
        return std::make_pair(w, tx);
    }

    inline std::pair<double, double> fineVoxSizeAndOrigin(unsigned subdivFactor = 2) const {
        const double w = mVoxSize / static_cast<double>(subdivFactor);
        const double tx = voxelOrigin() - static_cast<double>(subdivFactor - 1) * w * 0.5;
        return std::make_pair(w, tx);
    }

    void checkNonEmptyPrimalGrid() const {
        TORCH_CHECK(mGrid.data(), "Empty primal grid");
    }

    void setTransform(double voxSize, double voxOrigin) {
        const double w = voxSize;
        const double tx = voxOrigin;
        const double invW = 1.0 / w;

        mVoxSize = voxSize;

        mPrimalTransform = fvdb::VoxelCoordTransform(invW, static_cast<double>(-tx / w));
        mDualTransform = fvdb::VoxelCoordTransform(invW, (-tx / w + static_cast<double>(0.5)));
    }
};


template <typename BufferType>
torch::Tensor SparseFeatureIndexGrid<BufferType>::pointsInGrid(const torch::Tensor& points) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(points, mDeviceIndex);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(points.size(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));

    auto opts = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    torch::Tensor outMask = torch::empty({points.size(0)}, opts);
    fvdb::dispatchPointsInGrid<BufferType>(mGrid, mPrimalTransform, points, outMask);
    return outMask;
}


template <typename BufferType>
torch::Tensor SparseFeatureIndexGrid<BufferType>::ijkToIndex(const torch::Tensor& coords) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(coords, mDeviceIndex);
    TORCH_CHECK_TYPE(at::isIntegralType(coords.scalar_type(), false), "coords must have an integer type");
    TORCH_CHECK(coords.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(coords.dim()) + " dimensions");
    TORCH_CHECK(coords.size(0) > 0, "Empty tensor (coords)");
    TORCH_CHECK(coords.size(1) == 3,
                "Expected 3 dimensional coords but got points.shape[1] = " +
                std::to_string(coords.size(1)));

    auto opts = torch::TensorOptions().dtype(torch::kLong).device(coords.device());
    torch::Tensor outIndex = torch::empty({coords.size(0)}, opts);
    fvdb::dispatchIjkToIndex<BufferType>(mGrid, coords, outIndex);
    return outIndex;
}


template <typename BufferType>
void SparseFeatureIndexGrid<BufferType>::buildFromPointsWithPadding(const torch::Tensor& points,
                                                                    const nanovdb::Coord& padMin,
                                                                    const nanovdb::Coord& padMax) {
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(points.numel() > 0, "Cannot build empty grid, points tensor was empty.");
    TORCH_CHECK(points.size(0) > 0, "Cannot build empty grid, points tensor was empty.");
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));

    // We always need to build on the CPU for now unfortunately
    torch::Tensor cpuPoints = points;
    if (points.is_cuda()) {
        cpuPoints = points.cpu();
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "buildPaddedGridFromPoints", [&]() {
        using AccT = fvdb::PytorchTensorReadAccessor<scalar_t, 2>;
        mGrid = fvdb::buildPaddedGridFromPoints<BufferType, AccT>(
            fvdb::PytorchTensorReadAccessor<scalar_t, 2>(cpuPoints.accessor<scalar_t, 2>()),
            mPrimalTransform, padMin, padMax);
    });

    fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
}


template <typename BufferType>
void SparseFeatureIndexGrid<BufferType>::buildFromPointsNearestNeighbors(const torch::Tensor& points) {
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(points.numel() > 0, "Cannot build empty grid, points tensor was empty.");
    TORCH_CHECK(points.size(0) > 0, "Cannot build empty grid, points tensor was empty.");
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));

    // We always need to build on the CPU for now unfortunately
    torch::Tensor cpuPoints = points;
    if (points.is_cuda()) {
        cpuPoints = points.cpu();
    }

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "buildNearestNeighborGridFromPoints", [&]() {
        using AccT = fvdb::PytorchTensorReadAccessor<scalar_t, 2>;
        mGrid = fvdb::buildNearestNeighborGridFromPoints<BufferType, AccT>(
            fvdb::PytorchTensorReadAccessor<scalar_t, 2>(cpuPoints.accessor<scalar_t, 2>()),
            mPrimalTransform);
    });

    fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
}


template <typename BufferType>
void SparseFeatureIndexGrid<BufferType>::buildFromVoxelCoordinates(const torch::Tensor& coords,
                                                                   const nanovdb::Coord& padMin,
                                                                   const nanovdb::Coord& padMax) {
    TORCH_CHECK_TYPE(at::isIntegralType(coords.scalar_type(), false), "coords must have an integer type");
    TORCH_CHECK(coords.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(coords.dim()) + " dimensions");
    TORCH_CHECK(coords.numel() > 0, "Cannot build empty grid, coords tensor was empty.");
    TORCH_CHECK(coords.size(0) > 0, "Cannot build empty grid, coords tensor was empty.");
    TORCH_CHECK(coords.size(1) == 3,
                "Expected 3 dimensional coords but got points.shape[1] = " +
                std::to_string(coords.size(1)));

    // We always need to build on the CPU for now unfortunately
    torch::Tensor cpuCoords = coords;
    if (coords.is_cuda()) {
        cpuCoords = coords.cpu();
    }

    AT_DISPATCH_INTEGRAL_TYPES(coords.scalar_type(), "buildPaddedGridFromCoords", [&]() {
        using AccT = fvdb::PytorchTensorReadAccessor<scalar_t, 2>;
        mGrid = fvdb::buildPaddedGridFromCoords<BufferType, AccT>(
            fvdb::PytorchTensorReadAccessor<scalar_t, 2>(cpuCoords.accessor<scalar_t, 2>()),
            mPrimalTransform, padMin, padMax);
    });

    fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
}


template <typename BufferType>
void SparseFeatureIndexGrid<BufferType>::buildCoarseFromFineGrid(const SparseFeatureIndexGrid<BufferType>& fineGrid, unsigned branchFactor) {
    std::pair<double, double> wt = fineGrid.coarseVoxSizeAndOrigin(branchFactor);
    setTransform(wt.first, wt.second);

    mGrid = fvdb::buildCoarseGridFromFineGrid(fineGrid.mGrid, branchFactor);
    fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
}


template <typename BufferType>
void SparseFeatureIndexGrid<BufferType>::buildFineFromCoarseGrid(const SparseFeatureIndexGrid<BufferType>& coarseGrid, const torch::Tensor& subdivMask, unsigned subdivFactor) {
    TORCH_CHECK(subdivMask.sizes().size() == 1, "subdivision mask must have 1 dimension");
    TORCH_CHECK(subdivMask.size(0) == 0 || subdivMask.size(0) == coarseGrid.numVoxels(),
                "subdivision mask must be either empty tensor or have one entry per voxel");
    std::pair<double, double> wt = coarseGrid.fineVoxSizeAndOrigin(subdivFactor);
    setTransform(wt.first, wt.second);

    // We always need to build on the CPU for now unfortunately
    torch::Tensor cpuMask = subdivMask;
    if (subdivMask.is_cuda()) {
        cpuMask = subdivMask.cpu();
    }

    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, subdivMask.scalar_type(), "buildFineFromCoarseGrid", [&]() {
        mGrid = fvdb::buildFineGridFromCoarseGrid<BufferType, fvdb::PytorchTensorReadAccessor<scalar_t, 1>>(
            coarseGrid.mGrid,
            fvdb::PytorchTensorReadAccessor<scalar_t, 1>(cpuMask.accessor<scalar_t, 1>()),
            subdivFactor);
    });

    fvdb::maybeUploadToGPU<BufferType>(mGrid, mDeviceIndex);
}


template <typename BufferType>
torch::Tensor SparseFeatureIndexGrid<BufferType>::gridToWorld(const torch::Tensor& voxCoords) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(voxCoords, mDeviceIndex);
    TORCH_CHECK_TYPE(voxCoords.is_floating_point(), "vox_coords must be have a floating point type");

    torch::Tensor voxCoordsReshape = voxCoords.view({-1, 3});

    torch::Tensor outCoords = fvdb::TransformPoints<BufferType>::apply(mPrimalTransform, voxCoordsReshape, true)[0];

    return outCoords.reshape_as(voxCoordsReshape);
}


template <typename BufferType>
torch::Tensor SparseFeatureIndexGrid<BufferType>::worldToGrid(const torch::Tensor& worldCoords) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(worldCoords, mDeviceIndex);
    TORCH_CHECK_TYPE(worldCoords.is_floating_point(), "world_coords must be have a floating point type");
    torch::Tensor worldCoordsReshape = worldCoords.view({-1, 3});

    torch::Tensor outCoords = fvdb::TransformPoints<BufferType>::apply(mPrimalTransform, worldCoordsReshape, false)[0];
    return outCoords.reshape_as(worldCoords);
}


template <typename BufferType>
torch::Tensor SparseFeatureIndexGrid<BufferType>::activeGridCoords() const {
    checkNonEmptyPrimalGrid();
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(fvdb::bufferToTorchDevice<BufferType>(mDeviceIndex));

    torch::Tensor outPrimalCoord = torch::empty({numVoxels(), 3}, opts);
    fvdb::dispatchActiveGridCoords(mGrid, outPrimalCoord);

    return outPrimalCoord;
}


template <typename BufferType>
std::vector<torch::Tensor> SparseFeatureIndexGrid<BufferType>::splatTrilinear(
    const torch::Tensor& points, const torch::Tensor& pointPrimalData, bool returnCounts) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(points, mDeviceIndex);
    fvdb::checkDevice<BufferType>(pointPrimalData, mDeviceIndex);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(pointPrimalData.is_floating_point(), "points_primal_data must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == pointPrimalData.dtype(), "all tensors must have the same type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(pointPrimalData.dim() >= 2, std::string("Expected points_primal_data to have at least 2 dimensions ") +
                                            "(shape (n, ..., d)) but got " +
                                            std::to_string(pointPrimalData.dim()) + "dimensions");
    TORCH_CHECK(points.size(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(pointPrimalData.size(0) > 0, "Empty tensor (points_primal_data)");
    TORCH_CHECK(points.size(0) == pointPrimalData.size(0),
                "points and points_primal_data must have the same size in dimension 0 but got " +
                std::to_string(points.size(0)) + " and " + std::to_string(pointPrimalData.size(0)));
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));

    return fvdb::SplatNanovdbGrid<BufferType>::apply(
        mGrid, mPrimalTransform, points, pointPrimalData, returnCounts);
}


template <typename BufferType>
std::vector<torch::Tensor> SparseFeatureIndexGrid<BufferType>::sampleTrilinear(const torch::Tensor& points,
                                                                   const torch::Tensor& primalData,
                                                                   bool returnGrad) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(points, mDeviceIndex);
    fvdb::checkDevice<BufferType>(primalData, mDeviceIndex);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(primalData.is_floating_point(), "primal_data must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == primalData.dtype(), "all tensors must have the same type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(primalData.dim() >= 2, std::string("Expected primal_data to have at least 2 dimensions ") +
                                     "(shape (num_corners, ..., d)) but got " +
                                     std::to_string(primalData.dim()) + "dimensions");
    TORCH_CHECK(points.size(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(primalData.size(0) > 0, "Empty tensor (primal_data)");
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));
    TORCH_CHECK(primalData.size(0) == numVoxels(), "primal_data must have one value per voxel");


    return fvdb::SampleNanovdbGrid<BufferType>::apply(mGrid, mPrimalTransform, points, primalData, returnGrad, fvdb::SampleMode::TRILINEAR);
}


template <typename BufferType>
std::vector<torch::Tensor> SparseFeatureIndexGrid<BufferType>::sampleBezier(const torch::Tensor& points,
                                                                            const torch::Tensor& primalData,
                                                                            bool returnGrad) const {
    checkNonEmptyPrimalGrid();
    fvdb::checkDevice<BufferType>(points, mDeviceIndex);
    fvdb::checkDevice<BufferType>(primalData, mDeviceIndex);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(primalData.is_floating_point(), "primal_data must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == primalData.dtype(), "all tensors must have the same type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(primalData.dim() >= 2, std::string("Expected primal_data to have at least 2 dimensions ") +
                                       "(shape (num_corners, ..., d)) but got " +
                                       std::to_string(primalData.dim()) + "dimensions");
    TORCH_CHECK(points.size(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(primalData.size(0) > 0, "Empty tensor (primal_data)");
    TORCH_CHECK(points.size(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                std::to_string(points.size(1)));
    TORCH_CHECK(primalData.size(0) == numVoxels(), "primal_data must have one value per voxel");


    return fvdb::SampleNanovdbGrid<BufferType>::apply(mGrid, mPrimalTransform, points, primalData, returnGrad, fvdb::SampleMode::BEZIER);
}
