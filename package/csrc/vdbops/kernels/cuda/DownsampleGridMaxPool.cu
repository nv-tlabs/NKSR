#include "Kernels.h"

#include <c10/cuda/CUDAException.h>

#include "utils/cuda/Math.cuh"
#include "utils/PytorchDeviceBuffer.h"

constexpr int AC_NUM_BLOCK = fvdb::IndexTree::LeafNodeType::NUM_VALUES;
constexpr int AC_EACH_CHUNK = (1 << 0);
constexpr int AC_NUM_CHUNKS = AC_NUM_BLOCK / AC_EACH_CHUNK;

template <typename Dtype>
__global__ void downsampleGridMaxPool(const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                                      const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                                      const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> fineData,
                                      torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outCoarseData,
                                      uint32_t poolingFactor) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t l = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (totalIdx >= coarseGrid->tree().nodeCount(0) * AC_NUM_CHUNKS || l >= outCoarseData.size(1)) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = coarseGrid->tree().template getFirstNode<0>()[li];

    auto fineGridAcc = fineGrid->getAccessor();

    #pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(lo);
        const nanovdb::Coord fineIjk0(coarseIjk[0] * poolingFactor,
                                      coarseIjk[1] * poolingFactor,
                                      coarseIjk[2] * poolingFactor);
        if (leaf.isActive(lo)) {
            const int64_t coarseIndex = leaf.getValue(lo) - static_cast<int64_t>(1);
            outCoarseData[coarseIndex][l] = -INFINITY;

            for (unsigned i = 0; i < poolingFactor; i += 1) {
                for (unsigned j = 0; j < poolingFactor; j += 1) {
                    for (unsigned k = 0; k < poolingFactor; k += 1) {
                        nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                        const uint64_t fineIndex = fineGridAcc.getValue(fineIjk);
                        if (fineIndex == 0) {
                            continue;
                        }
                        const Dtype currentValue = outCoarseData[coarseIndex][l];

                        outCoarseData[coarseIndex][l] =
                            fmax<Dtype>(fineData[fineIndex - 1][l], currentValue);
                    }
                }
            }
        }
    }
}


template <typename Dtype>
__global__ void downsampleGridMaxPoolGrad(const nanovdb::NanoGrid<nanovdb::ValueIndex>* coarseGrid,
                                          const nanovdb::NanoGrid<nanovdb::ValueIndex>* fineGrid,
                                          const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> fineData,
                                          const torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> coarseGradOut,
                                          torch::PackedTensorAccessor32<Dtype, 2, torch::RestrictPtrTraits> outFineGradIn,
                                          uint32_t poolingFactor) {

    const int32_t totalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int32_t l = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (totalIdx >= coarseGrid->tree().nodeCount(0) * AC_NUM_CHUNKS || l >= coarseGradOut.size(1)) {
        return;
    }
    const int32_t li = totalIdx / AC_NUM_CHUNKS;
    const int32_t cidx = totalIdx % AC_NUM_CHUNKS;

    const fvdb::IndexTree::LeafNodeType& leaf = coarseGrid->tree().template getFirstNode<0>()[li];

    auto fineGridAcc = fineGrid->getAccessor();

    #pragma unroll
    for (uint32_t loid = 0; loid < AC_EACH_CHUNK; loid += 1) {
        uint32_t lo = cidx * AC_EACH_CHUNK + loid;
        const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(lo);
        const nanovdb::Coord fineIjk0(coarseIjk[0] * poolingFactor,
                                      coarseIjk[1] * poolingFactor,
                                      coarseIjk[2] * poolingFactor);
        if (leaf.isActive(lo)) {
            Dtype maxValue = -INFINITY;
            int64_t maxIndex = -1;

            for (unsigned i = 0; i < poolingFactor; i += 1) {
                for (unsigned j = 0; j < poolingFactor; j += 1) {
                    for (unsigned k = 0; k < poolingFactor; k += 1) {
                        nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                        const uint64_t fineIndex = fineGridAcc.getValue(fineIjk);
                        if (fineIndex == 0) {
                            continue;
                        }

                        const Dtype fineValue = fineData[fineIndex - 1][l];
                        if (fineValue > maxValue) {
                            maxIndex = fineIndex;
                            maxValue = fineValue;
                        }
                    }
                }
            }

            if (maxIndex >= 0) {
                outFineGradIn[maxIndex - 1][l] = coarseGradOut[leaf.getValue(lo) - 1][l];
            }
        }
    }
}


namespace fvdb {

template <>
void dispatchDownsampleGridMaxPool<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& fineGridHdl,
                                                        const nanovdb::GridHandle<PytorchDeviceBuffer>& coarseGridHdl,
                                                        torch::Tensor& fineData,
                                                        torch::Tensor& outCoarseData,
                                                        unsigned downsamplingFactor,
                                                        unsigned nThreadsX,
                                                        unsigned nThreadsY) {
    const auto* fineGrid = fineGridHdl.deviceGrid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* coarseGrid = coarseGridHdl.deviceGrid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t VOXCOUNT = AC_NUM_CHUNKS * coarseGridHdl.grid<nanovdb::ValueIndex>()->tree().nodeCount(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(VOXCOUNT, NTHREADSX);

    const int64_t DCOUNT = outCoarseData.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(fineData.scalar_type(), "downsampleGridMaxPool", [&]() {
        downsampleGridMaxPool<scalar_t><<<nblocks, nthreads>>>(fineGrid, coarseGrid,
                                                               fineData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                               outCoarseData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                               downsamplingFactor);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}


template <>
void dispatchDownsampleGridMaxPoolGrad<PytorchDeviceBuffer>(const nanovdb::GridHandle<PytorchDeviceBuffer>& coarseGridHdl,
                                                            const nanovdb::GridHandle<PytorchDeviceBuffer>& fineGridHdl,
                                                            const torch::Tensor& fineData,
                                                            const torch::Tensor& coarseGradOut,
                                                            torch::Tensor& outFineGradIn,
                                                            unsigned poolingFactor,
                                                            unsigned nThreadsX,
                                                            unsigned nThreadsY) {
    const auto* fineGrid = fineGridHdl.deviceGrid<nanovdb::ValueIndex>();
    if (!fineGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const auto* coarseGrid = coarseGridHdl.deviceGrid<nanovdb::ValueIndex>();
    if (!coarseGrid) {
        throw std::runtime_error("Failed to get pointer to nanovdb index grid");
    }

    const int64_t VOXCOUNT = AC_NUM_CHUNKS * coarseGridHdl.grid<nanovdb::ValueIndex>()->tree().nodeCount(0);
    const int64_t NTHREADSX = nThreadsX;
    const int64_t NBLOCKSX = GET_BLOCKS(VOXCOUNT, NTHREADSX);

    const int64_t DCOUNT = coarseGradOut.size(1);
    const int64_t NTHREADSY = nThreadsY;
    const int64_t NBLOCKSY = GET_BLOCKS(DCOUNT, NTHREADSY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(NTHREADSX, NTHREADSY, 1);

    AT_DISPATCH_FLOATING_TYPES(coarseGradOut.scalar_type(), "downsampleGridMaxPoolGrad", [&]() {
        downsampleGridMaxPoolGrad<scalar_t><<<nblocks, nthreads>>>(coarseGrid, fineGrid,
                                                                   fineData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                   coarseGradOut.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                   outFineGradIn.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                   poolingFactor);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

} // namespace fvdb