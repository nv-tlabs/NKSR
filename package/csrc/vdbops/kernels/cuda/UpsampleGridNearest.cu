#include "Kernels.h"

#include <c10/cuda/CUDAException.h>
#include <THC/THCAtomics.cuh>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"


constexpr int AC_NUM_BLOCK = fvdb::IndexTree::LeafNodeType::NUM_VALUES;
constexpr int AC_EACH_CHUNK = (1 << 2);
constexpr int AC_NUM_CHUNKS = AC_NUM_BLOCK / AC_EACH_CHUNK;

template <typename Dtype>
__global__ void upsampleGridNearest(const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                                    const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                                    const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> coarseData,
                                    torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outFineData,
                                    uint32_t upsamplingFactor) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t i = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (totalIdx >= fineGrid->tree().nodeCount(0) * AC_NUM_CHUNKS || i >= outFineData.size(1)) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = fineGrid->tree().template getFirstNode<0>()[li];

    auto coarseGridAcc = coarseGrid->getAccessor();

    #pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (leaf.isActive(lo)) {
            const nanovdb::Coord fineIjk = leaf.offsetToGlobalCoord(lo);
            const nanovdb::Coord coarseIjk = nanovdb::Vec3<Dtype>((Dtype) fineIjk[0] / (Dtype) upsamplingFactor,
                                                                  (Dtype) fineIjk[1] / (Dtype) upsamplingFactor,
                                                                  (Dtype) fineIjk[2] / (Dtype) upsamplingFactor).floor();
            const int64_t coarseIndex = coarseGridAcc.getValue(coarseIjk) - static_cast<int64_t>(1);
            if (coarseIndex < 0) {
                continue;
            }
            outFineData[leaf.getValue(lo) - 1][i] = coarseData[coarseIndex][i];
        }
    }
}


template <typename Dtype>
__global__ void upsampleGridNearestGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                                        const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                                        const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> fineData,
                                        torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outCoarseData,
                                        uint32_t upsamplingFactor) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t i = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (totalIdx >= fineGrid->tree().nodeCount(0) * AC_NUM_CHUNKS || i >= outCoarseData.size(1)) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = fineGrid->tree().template getFirstNode<0>()[li];

    auto coarseGridAcc = coarseGrid->getAccessor();

    #pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        if (leaf.isActive(lo)) {
            const nanovdb::Coord fineIjk = leaf.offsetToGlobalCoord(lo);
            const nanovdb::Coord coarseIjk = nanovdb::Vec3<Dtype>((Dtype) fineIjk[0] / (Dtype) upsamplingFactor,
                                                                  (Dtype) fineIjk[1] / (Dtype) upsamplingFactor,
                                                                  (Dtype) fineIjk[2] / (Dtype) upsamplingFactor).floor();
            const int64_t coarseIndex = coarseGridAcc.getValue(coarseIjk) - static_cast<int64_t>(1);
            if (coarseIndex < 0) {
                continue;
            }
            gpuAtomicAddNoReturn(&outCoarseData[coarseIndex][i], fineData[leaf.getValue(lo) - 1][i]);
        }
    }
}




namespace fvdb {

template <>
void dispatchUpsampleGridNearest<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& coarseGridHdl,
                                                      const nanovdb::GridHandle<PytorchDeviceBuffer>& fineGridHdl,
                                                      torch::Tensor& coarseData,
                                                      torch::Tensor& outFineData,
                                                      unsigned upsamplingFactor,
                                                      unsigned nThreadsX,
                                                      unsigned nThreadsY) {
    const auto* coarseGrid = coarseGridHdl.template deviceGrid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* fineGrid = fineGridHdl.template deviceGrid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t VOXCOUNT = AC_NUM_CHUNKS * fineGridHdl.grid<nanovdb::ValueIndex>()->tree().nodeCount(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(VOXCOUNT, NTHREADSX);

    const int64_t DCOUNT = outFineData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(coarseData.scalar_type(), "upsampleGridNearest", [&]() {
        upsampleGridNearest<scalar_t><<<nblocks, nthreads>>>(coarseGrid, fineGrid,
                                                             coarseData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                             outFineData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                             upsamplingFactor);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}


template <>
void dispatchUpsampleGridNearestGrad<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& fineGridHdl,
                                                          const nanovdb::GridHandle<PytorchDeviceBuffer>& coarseGridHdl,
                                                          torch::Tensor& fineData,
                                                          torch::Tensor& outCoarseData,
                                                          unsigned upsamplingFactor,
                                                          unsigned nThreadsX,
                                                          unsigned nThreadsY) {
    const auto* fineGrid = fineGridHdl.template grid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* coarseGrid = coarseGridHdl.template grid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t VOXCOUNT = AC_NUM_CHUNKS * fineGridHdl.grid<nanovdb::ValueIndex>()->tree().nodeCount(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(VOXCOUNT, NTHREADSX);

    const int64_t DCOUNT = outCoarseData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(fineData.scalar_type(), "upsampleGridNearestGrad", [&]() {
        upsampleGridNearestGrad<scalar_t><<<nblocks, nthreads>>>(fineGrid, coarseGrid,
                                                                 fineData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                 outCoarseData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                 upsamplingFactor);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

} // namespace fvdb