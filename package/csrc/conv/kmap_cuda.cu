#include "conv.h"
#include "utils/cuda/Math.cuh"

constexpr int AC_NUM_BLOCK = fvdb::IndexTree::LeafNodeType::NUM_VALUES;
constexpr int AC_EACH_CHUNK = 4;
constexpr int AC_NUM_CHUNKS = AC_NUM_BLOCK / AC_EACH_CHUNK;

/**
 * Build Kernel map for convolution
 */

__global__ void convolutionKernelMapKernel(
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* source,
        const nanovdb::NanoGrid<nanovdb::ValueIndex>* target,
        int kernelStart, int kernel, int stride,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> kmap) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t nodeCount = target->tree().nodeCount(0);

    if (totalIdx >= nodeCount * AC_NUM_CHUNKS) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = target->tree().template getFirstNode<0>()[li];
    auto sourceAcc = source->getAccessor();

#pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (leaf.isActive(lo)) {
            nanovdb::Coord tCoord = leaf.offsetToGlobalCoord(lo);
            nanovdb::Coord sCoord(tCoord.x() * stride, tCoord.y() * stride, tCoord.z() * stride);
            int kIdx = 0;
#pragma unroll
            for (int kx = kernelStart; kx < kernelStart + kernel; ++kx) {
                for (int ky = kernelStart; ky < kernelStart + kernel; ++ky) {
                    for (int kz = kernelStart; kz < kernelStart + kernel; ++kz, ++kIdx) {
                        nanovdb::Coord sOffset = sCoord + nanovdb::Coord(kx, ky, kz);
                        if (sourceAcc.isActive(sOffset)) {
                            kmap[leaf.getValue(lo) - 1][kIdx] = sourceAcc.getValue(sOffset) - 1;
                        } else {
                            kmap[leaf.getValue(lo) - 1][kIdx] = -1;
                        }
                    }
                }
            }
        }
    }
}

template<>
torch::Tensor convolutionKernelMap(const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer> &source,
                                   const SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer> &target,
                                   int kernel,
                                   unsigned nThreadsX) {

    TORCH_CHECK(source.voxelSize() <= target.voxelSize(), "source must have smaller voxel size than target!");
    TORCH_CHECK(kernel > 0, "kernel must be positive integer!");

    int stride = (int) std::round(target.voxelSize() / source.voxelSize());
    int kernelVolume = kernel * kernel * kernel;
    int kernelStart = (int) std::floor(-kernel / 2.0 + 1);

    torch::Tensor kmap = torch::empty({target.numVoxels(), kernelVolume}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(target.device()));
    const auto* sourceGrid = source.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();
    const auto* targetGrid = target.nanovdbGrid().deviceGrid<nanovdb::ValueIndex>();

    const int64_t PCOUNT = target.nanovdbGrid().grid<nanovdb::ValueIndex>()->tree().nodeCount(0) * AC_NUM_CHUNKS;
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    convolutionKernelMapKernel<<<nblocks, nthreads>>>(
            sourceGrid, targetGrid, kernelStart, kernel, stride,
            kmap.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return kmap;
}

