namespace fvdb {

template <typename Dtype>
static __device__ Dtype fmax(Dtype a, Dtype b);



template <typename Dtype>
static __device__ Dtype fmin(Dtype a, Dtype b);


template <typename Dtype>
static __device__ Dtype ceiling(Dtype x);


// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
static int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS);


template <>
__device__ float fmin<float>(float a, float b) {
    return fminf(a, b);
}

template <>
__device__ double fmin<double>(double a, double b) {
    return fmin(a, b);
}


template <>
__device__ float fmax<float>(float a, float b) {
    return fmaxf(a, b);
}

template <>
__device__ double fmax<double>(double a, double b) {
    return fmax(a, b);
}


template <>
__device__ float ceiling<float>(float x) {
    return ceilf(x);
}

template <>
__device__ double ceiling<double>(double x) {
    return ceil(x);
}


int static GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block) {
    TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
    constexpr int64_t max_int = std::numeric_limits<int>::max();

    // Round up division for positive number that cannot cause integer overflow
    auto block_num = (N - 1) / max_threads_per_block + 1;
    TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

    return static_cast<int>(block_num);
}

} // namespace fvdb