#include "Kernels.h"

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


__global__ void activeGridCoords(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                                 torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> outGridCoords) {
    const int32_t li = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (li >= gpuGrid->tree().nodeCount(0)) {
        return;
    }

    const fvdb::IndexTree::LeafNodeType& leaf = gpuGrid->tree().template getFirstNode<0>()[li];

    #pragma unroll
    for (uint32_t lo = 0; lo < fvdb::IndexTree::LeafNodeType::NUM_VALUES; lo += 1) {
        if (leaf.isActive(lo)) {
            nanovdb::Coord ijk = leaf.offsetToGlobalCoord(lo);
            #pragma unroll
            for (int k = 0; k < 3; k += 1) {
                outGridCoords[leaf.getValue(lo) - 1][k] = ijk[k];
            }
        }
    }
}




namespace fvdb {

template<>
void dispatchActiveGridCoords<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                                   torch::Tensor& outGridCoords,
                                                   unsigned nThreadsX) {
    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = gridBuf.template grid<nanovdb::ValueIndex>()->tree().nodeCount(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    activeGridCoords<<<nblocks, nthreads>>>(
        gpuGrid, outGridCoords.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb