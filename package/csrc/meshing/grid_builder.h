#include <SparseFeatureIndexGrid.h>
#include "../common/iter_util.h"

/**
 * Conformal Flattened Grid Builder (will use CPU throughout)
 */
template <typename BufferType>
std::shared_ptr<SparseFeatureIndexGrid<BufferType>> buildFlattenedGrid(
        const std::shared_ptr<SparseFeatureIndexGrid<BufferType>>& thisGrid,
        const torch::optional<std::shared_ptr<SparseFeatureIndexGrid<BufferType>>>& childGrid,
        bool conforming) {

    torch::Tensor childMask = torch::zeros(thisGrid->numVoxels(),
                                           torch::TensorOptions().device(torch::kCPU).dtype(torch::kBool));
    auto childMaskAcc = childMask.accessor<bool, 1>();
    auto thisGridBuf = thisGrid->nanovdbGrid().template grid<nanovdb::ValueIndex>();
    auto thisGridAcc = thisGridBuf->getAccessor();

    if (childGrid) {
        // Check if any voxel contains child.
        auto childGridBuf = (*childGrid)->nanovdbGrid().template grid<nanovdb::ValueIndex>();
        auto childGridAcc = childGridBuf->getAccessor();
        for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(thisGridBuf->tree()); it.isValid(); it++) {
            for (auto jt = OctChildrenIterator(it->first << 1); jt.isValid(); ++jt) {
                if (childGridAcc.isActive(*jt)) {
                    childMaskAcc[it->second] = true;
                    break;
                }
            }
        }
    }

    // Build new grid, without those who are marked to have kids.
    auto newSVH = std::make_shared<SparseFeatureIndexGrid<BufferType>>(
            thisGrid->voxelSize(), thisGrid->voxelOrigin());
    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();

    if (!conforming) {
        // For non-conforming grid, simply remove those with kids
        for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(thisGridBuf->tree()); it.isValid(); it++) {
            if (!childMaskAcc[it->second]) {
                proxyGridAccessor.setValue(it->first, 1);
            }
        }
    } else {
        // For conforming grid, should add all 8 siblings
        for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(thisGridBuf->tree()); it.isValid(); it++) {
            for (auto ct = OctChildrenIterator(it->first); ct.isValid(); ++ct) {
                int64_t vidx = voxelIndex(thisGridAcc, *ct);
                if (vidx == -1 || !childMaskAcc[vidx]) {
                    proxyGridAccessor.setValue(*ct, 1);
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    auto newHandle = idxGridBuilder.getHandle<BufferType>();
    newSVH->buildFromNanoVDBGridHandle(newHandle);

    return newSVH;
}

/**
 * Joint dual grid Builder from the corners of the primal grids (will use CPU throughout)
 */
template <typename BufferType>
std::shared_ptr<SparseFeatureIndexGrid<BufferType>> buildJointDualGrid(
        const std::vector<std::shared_ptr<SparseFeatureIndexGrid<BufferType>>>& primalGrids) {

    double voxelSize = primalGrids[0]->voxelSize();
    double voxelOrigin = primalGrids[0]->voxelOrigin();

    auto newSVH = std::make_shared<SparseFeatureIndexGrid<BufferType>>(
            voxelSize, voxelOrigin - 0.5 * voxelSize);
    auto dualTransform = newSVH->primalTransform();
    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();

    for (auto pGrid : primalGrids) {
        auto pGridBuf = pGrid->nanovdbGrid().template grid<nanovdb::ValueIndex>();
        auto pTransform = pGrid->primalTransform();
        for (auto it = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(pGridBuf->tree()); it.isValid(); it++) {
            for (int ax = -1; ax < 2; ax += 2) {
                for (int ay = -1; ay < 2; ay += 2) {
                    for (int az = -1; az < 2; az += 2) {
                        nanovdb::Vec3d w = pTransform.template applyInv<double>(nanovdb::Vec3d(
                                (double) it->first[0] + ax * 0.5,
                                (double) it->first[1] + ay * 0.5,
                                (double) it->first[2] + az * 0.5));
                        nanovdb::Coord dc = dualTransform.template apply<double>(w).round();
                        proxyGridAccessor.setValue(dc, 1);
                    }
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    auto newHandle = idxGridBuilder.getHandle<BufferType>();
    newSVH->buildFromNanoVDBGridHandle(newHandle);

    return newSVH;
}
