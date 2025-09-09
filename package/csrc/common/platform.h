#pragma once

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __host__ __device__  // for CUDA device code
#else
#define _CPU_AND_GPU_CODE_
#endif

#if defined(__CUDACC__)
#define _CPU_AND_GPU_CODE_TEMPLATE_ __host__ __device__ // for CUDA device code
#else
#define _CPU_AND_GPU_CODE_TEMPLATE_
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CONSTANT_ __constant__	// for CUDA device code
#else
#define _CPU_AND_GPU_CONSTANT_ const
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#include <THC/THCAtomics.cuh>

template <typename ScalarT>
static inline __device__ void atomicAdd(ScalarT* addr, ScalarT value) {
    gpuAtomicAddNoReturn(addr, value);
}
#else
template <typename ScalarT>
static inline void atomicAdd(ScalarT* addr, ScalarT value) {
    *addr += value;
}
#endif

