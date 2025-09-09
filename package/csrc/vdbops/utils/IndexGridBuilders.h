#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IndexGridBuilder.h>

#include "ActiveVoxelIterator.h"
#include "VoxelCoordTransform.h"
#include "Printing.h"


namespace fvdb {

using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;


template <typename BufferType>
nanovdb::GridHandle<BufferType> buildCoarseGridFromFineGrid(const nanovdb::GridHandle<BufferType>& fineGridHdl,
                                                            const unsigned branchingFactor) {
    const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid = fineGridHdl.template grid<nanovdb::ValueIndex>(0);
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }
    const IndexTree& fineTree = fineGrid->tree();
    // const uint64_t numFineVoxels = fineTree.activeVoxelCount();

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();

    for (auto it = ActiveVoxelIterator<IndexTree>(fineTree); it.isValid(); it++) {
        const nanovdb::Coord coarseIjk = (it->first.asVec3d() / (double) branchingFactor).floor();
        proxyGridAccessor.setValue(coarseIjk, 1);
    }

    auto handle = proxyGridBuilder.getHandle();
    auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}


template <typename BufferType, typename AccMaskT>
nanovdb::GridHandle<BufferType> buildFineGridFromCoarseGrid(const nanovdb::GridHandle<BufferType>& coarseGridHdl,
                                                            const AccMaskT& subdivMask,
                                                            const unsigned subdivisionFactor) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid = coarseGridHdl.template grid<nanovdb::ValueIndex>(0);
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }
    const IndexTree& coarseTree = coarseGrid->tree();

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();

    for (auto it = ActiveVoxelIterator<IndexTree, -1>(coarseTree); it.isValid(); it++) {
        const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor,
                                     it->first[1] * subdivisionFactor,
                                     it->first[2] * subdivisionFactor);

        if (subdivMask.rows() > 0 && !subdivMask.get(it->second)) {
            continue;
        }

        for (unsigned i = 0; i < subdivisionFactor; i += 1) {
            for (unsigned j = 0; j < subdivisionFactor; j += 1) {
                for (unsigned k = 0; k < subdivisionFactor; k += 1) {
                    const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                    proxyGridAccessor.setValue(fineIjk, 1);
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}



template <typename BufferType, int BMIN=-1, int BMAX=1>
nanovdb::GridHandle<BufferType> buildPaddedGridFromGrid(const nanovdb::GridHandle<BufferType>& baseGridHdl) {
    static_assert(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* baseGrid = baseGridHdl.template grid<nanovdb::ValueIndex>(0);
    if (!baseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();

    for (auto it = ActiveVoxelIterator<IndexTree>(baseGrid->tree()); it.isValid(); it++) {
        nanovdb::Coord ijk0 = it->first;
        for (int di = BMIN; di <= BMAX; di += 1) {
            for (int dj = BMIN; dj <= BMAX; dj += 1) {
                for (int dk = BMIN; dk <= BMAX; dk += 1) {
                    const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                    proxyGridAccessor.setValue(ijk, 1);
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    const nanovdb::Int32Grid* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}



template <typename BufferType, typename AccPoints>
nanovdb::GridHandle<BufferType> buildPaddedGridFromPoints(const AccPoints& points,
                                                          const VoxelCoordTransform& tx,
                                                          const nanovdb::Coord& bmin,
                                                          const nanovdb::Coord& bmax) {
    using ScalarT = typename AccPoints::ScalarT;

    static_assert(std::is_floating_point<ScalarT>::value, "Invalid type for points, must be floating point");

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();
    for (unsigned pi = 0; pi < points.rows(); pi += 1) {

        nanovdb::Coord ijk0 = tx.apply(points.get(pi, 0), points.get(pi, 1), points.get(pi, 2)).round();

        // Splat the normal to the 8 neighboring voxels
        for (int di = bmin[0]; di <= bmax[0]; di += 1) {
            for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
                for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                    const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                    proxyGridAccessor.setValue(ijk, 11);
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    const auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}



template <typename BufferType, typename AccPoints>
nanovdb::GridHandle<BufferType> buildNearestNeighborGridFromPoints(const AccPoints& points,
                                                                   const VoxelCoordTransform& tx) {
    using ScalarT = typename AccPoints::ScalarT;
    using Vec3T = typename nanovdb::Vec3<ScalarT>;

    static_assert(std::is_floating_point<ScalarT>::value, "Invalid type for points, must be floating point");

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();
    for (unsigned pi = 0; pi < points.rows(); pi += 1) {
        Vec3T ijk0 = tx.apply(points.get(pi, 0), points.get(pi, 1), points.get(pi, 2));
        nanovdb::Coord ijk000 = (ijk0 + Vec3T(-0.5, -0.5, -0.5)).round();
        nanovdb::Coord ijk001 = (ijk0 + Vec3T(-0.5, -0.5,  0.5)).round();
        nanovdb::Coord ijk010 = (ijk0 + Vec3T(-0.5,  0.5, -0.5)).round();
        nanovdb::Coord ijk011 = (ijk0 + Vec3T(-0.5,  0.5,  0.5)).round();
        nanovdb::Coord ijk100 = (ijk0 + Vec3T( 0.5, -0.5, -0.5)).round();
        nanovdb::Coord ijk101 = (ijk0 + Vec3T( 0.5, -0.5,  0.5)).round();
        nanovdb::Coord ijk110 = (ijk0 + Vec3T( 0.5,  0.5, -0.5)).round();
        nanovdb::Coord ijk111 = (ijk0 + Vec3T( 0.5,  0.5,  0.5)).round();

        proxyGridAccessor.setValue(ijk000, 11);
        proxyGridAccessor.setValue(ijk001, 11);
        proxyGridAccessor.setValue(ijk010, 11);
        proxyGridAccessor.setValue(ijk011, 11);
        proxyGridAccessor.setValue(ijk100, 11);
        proxyGridAccessor.setValue(ijk101, 11);
        proxyGridAccessor.setValue(ijk110, 11);
        proxyGridAccessor.setValue(ijk111, 11);
    }

    auto handle = proxyGridBuilder.getHandle();
    const auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}

template <typename BufferType, typename AccPoints>
nanovdb::GridHandle<BufferType> buildPaddedGridFromCoords(const AccPoints& coords,
                                                          const VoxelCoordTransform& tx,
                                                          const nanovdb::Coord& bmin,
                                                          const nanovdb::Coord& bmax) {
    using ScalarT = typename AccPoints::ScalarT;

    static_assert(std::is_integral<ScalarT>::value, "Invalid type for coords, must be integral");

    nanovdb::GridBuilder<int32_t> proxyGridBuilder;
    auto proxyGridAccessor = proxyGridBuilder.getAccessor();
    for (unsigned ci = 0; ci < coords.rows(); ci += 1) {

        nanovdb::Coord ijk0(coords.get(ci, 0), coords.get(ci, 1), coords.get(ci, 2));

        // Splat the normal to the 8 neighboring voxels
        for (int di = bmin[0]; di <= bmax[0]; di += 1) {
            for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
                for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                    const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                    proxyGridAccessor.setValue(ijk, 11);
                }
            }
        }
    }

    auto handle = proxyGridBuilder.getHandle();
    const auto* grid = handle.grid<int32_t>();
    nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
    return idxGridBuilder.getHandle<BufferType>();
}


template <typename BufferType, typename AccPoints, typename AccDepths, int BMIN=-1, int BMAX=1>
void buildPaddedHierarchyFromPoints(const AccPoints& points,
                                    const AccDepths& depths,
                                    unsigned maxDepth,
                                    const std::vector<VoxelCoordTransform>& tx,
                                    std::vector<nanovdb::GridHandle<BufferType>>& outGrids) {
    static_assert(BMIN <= BMAX, "BMIN must be less than BMAX");
    using ScalarT = typename AccPoints::ScalarT;
    using IndexT = typename AccDepths::ScalarT;

    static_assert(std::is_integral<IndexT>::value, "Invalid type for depths, must be integers");
    static_assert(std::is_floating_point<ScalarT>::value, "Invalid type for points, must be floats");

    std::vector<nanovdb::GridBuilder<int32_t>> proxyGridBuilders(maxDepth);
    using AccessorT = decltype(proxyGridBuilders[0].getAccessor()); // nanovdb::GridBuilder<int32_t>::ValueAccessor;

    std::vector<AccessorT> proxyGridAccessors;
    for (int i = 0; i < maxDepth; i += 1) {
        proxyGridAccessors.push_back(proxyGridBuilders[i].getAccessor());
    }

    for (unsigned pi = 0; pi < points.rows(); pi += 1) {
        IndexT depth = depths.get(pi);
        nanovdb::Coord ijk0 = tx[depth].apply(points.get(pi, 0), points.get(pi, 1), points.get(pi, 2)).round();

        for (int di = BMIN; di <= BMAX; di += 1) {
            for (int dj = BMIN; dj <= BMAX; dj += 1) {
                for (int dk = BMIN; dk <= BMAX; dk += 1) {
                    const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                    proxyGridAccessors[depth].setValue(ijk, 11);

                    // Make sure we handle the parent
                    if (depth < maxDepth - 1) {
                        nanovdb::Coord parent = (ijk.asVec3d() / 2.0).floor();
                        proxyGridAccessors[depth + 1].setValue(parent, 11);
                    }
                }
            }
        }
    }

    for (int i = 0; i < maxDepth; i += 1) {
        auto handle = proxyGridBuilders[i].getHandle();
        const auto* grid = handle.grid<int32_t>();
        nanovdb::IndexGridBuilder<int32_t> idxGridBuilder(*grid, false, false);
        outGrids.push_back(idxGridBuilder.getHandle<BufferType>());
    }
}

} // namespace fvdb