#include "conv.h"

/**
 * Build Kernel map for convolution
 */
template<>
torch::Tensor convolutionKernelMap(const SparseFeatureIndexGrid<nanovdb::HostBuffer> &source,
                                   const SparseFeatureIndexGrid<nanovdb::HostBuffer> &target,
                                   int kernel,
                                   unsigned nThreadsX) {

    TORCH_CHECK(source.voxelSize() <= target.voxelSize(), "source must have smaller voxel size than target!");
    TORCH_CHECK(kernel > 0, "kernel must be positive integer!");

    int stride = (int) std::round(target.voxelSize() / source.voxelSize());
    int kernelVolume = kernel * kernel * kernel;
    int kernelStart = (int) std::floor(-kernel / 2.0 + 1);

    torch::Tensor kmap = torch::empty({target.numVoxels(), kernelVolume}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(target.device()));
    auto kmapAcc = kmap.accessor<int, 2>();
    const auto* sourceGrid = source.nanovdbGrid().grid<nanovdb::ValueIndex>();
    const auto* targetGrid = target.nanovdbGrid().grid<nanovdb::ValueIndex>();

    auto sourceAcc = sourceGrid->getAccessor();

    for (auto tit = fvdb::ActiveVoxelIterator<fvdb::IndexTree, -1>(targetGrid->tree()); tit.isValid(); tit++) {
        nanovdb::Coord sCoord(tit->first.x() * stride, tit->first.y() * stride, tit->first.z() * stride);
        // Center kernel is in the middle -- allows for acceleration in Conv.
        int kIdx = 0;
        for (int kx = kernelStart; kx < kernelStart + kernel; ++kx) {
            for (int ky = kernelStart; ky < kernelStart + kernel; ++ky) {
                for (int kz = kernelStart; kz < kernelStart + kernel; ++kz, ++kIdx) {
                    nanovdb::Coord sOffset = sCoord + nanovdb::Coord(kx, ky, kz);
                    if (sourceAcc.isActive(sOffset)) {
                        kmapAcc[tit->second][kIdx] = voxelIndex(sourceAcc, sOffset);
                    } else {
                        kmapAcc[tit->second][kIdx] = -1;
                    }
                }
            }
        }
    }

    return kmap;
}
