#include "Kernels.h"

#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


template <typename Dtype>
__global__ void ijkToIndex(const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid,
                           const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> coords,
                           torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> outIndex) {

    const int32_t ci = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (ci >= coords.size(0)) {
        return;
    }

    auto acc = gpuGrid->getAccessor();

    const nanovdb::Coord vox(coords[ci][0], coords[ci][1], coords[ci][2]);
    if (acc.isActive(vox)) {
        outIndex[ci] = acc.getValue(vox) - 1;
    } else {
        outIndex[ci] = -1;
    }
}


namespace fvdb {

template <>
void dispatchIjkToIndex<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& gridBuf,
                                             const torch::Tensor& coords,
                                             torch::Tensor& outIndex,
                                             unsigned nThreadsX) {

    const nanovdb::NanoGrid<nanovdb::ValueIndex>* gpuGrid = gridBuf.template deviceGrid<nanovdb::ValueIndex>();
    if (!gpuGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t PCOUNT = coords.size(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(PCOUNT, NTHREADSX);

    dim3 nblocks(NBLOCKSX, 1, 1);
    dim3 nthreads(NTHREADSX, 1, 1);

    AT_DISPATCH_INTEGRAL_TYPES(coords.scalar_type(), "ijkToIndex", [&]() {
        ijkToIndex<scalar_t><<<nblocks, nthreads>>>(gpuGrid,
                                                    coords.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                    outIndex.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

} // namespace fvdb