#pragma once

#include <nanovdb/util/HostBuffer.h> // for BufferTraits
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>

#include <cuda_runtime_api.h> // for cudaMalloc/cudaMallocManaged/cudaFree

#include <c10/cuda/CUDACachingAllocator.h>



namespace fvdb {

// ----------------------------> PytorchDeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class PytorchDeviceBuffer
{
    uint64_t mSize; // total number of bytes for the NanoVDB grid.
    uint8_t *mCpuData, *mGpuData; // raw buffer for the NanoVDB grid.


public:
    PytorchDeviceBuffer(uint64_t size = 0, void* data = nullptr)
        : mSize(0)
        , mCpuData(nullptr)
        , mGpuData(nullptr)
    {
        this->init(size, data);
    }
    /// @brief Disallow copy-construction
    PytorchDeviceBuffer(const PytorchDeviceBuffer&) = delete;
    /// @brief Move copy-constructor
    PytorchDeviceBuffer(PytorchDeviceBuffer&& other) noexcept
        : mSize(other.mSize)
        , mCpuData(other.mCpuData)
        , mGpuData(other.mGpuData)
    {
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
    }
    /// @brief Disallow copy assignment operation
    PytorchDeviceBuffer& operator=(const PytorchDeviceBuffer&) = delete;
    /// @brief Move copy assignment operation
    PytorchDeviceBuffer& operator=(PytorchDeviceBuffer&& other) noexcept
    {
        clear();
        mSize = other.mSize;
        mCpuData = other.mCpuData;
        mGpuData = other.mGpuData;
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
        return *this;
    }
    /// @brief Destructor frees memory on both the host and device
    ~PytorchDeviceBuffer() { this->clear(); };

    void init(uint64_t size, void* data = nullptr);

    // @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    ///
    /// @warning Note that the pointer can be NULL is the allocator was not initialized!
    uint8_t* data() const { return mCpuData; }
    uint8_t* deviceData() const { return mGpuData; }

    /// @brief Copy grid from the CPU/host to the GPU/device. If @c sync is false the memory copy is asynchronous!
    ///
    /// @note This will allocate memory on the GPU/device if it is not already allocated
    void deviceUpload(void* stream = 0, bool sync = true);

    /// @brief Copy grid from the GPU/device to the CPU/host. If @c sync is false the memory copy is asynchronous!
    void deviceDownload(void* stream = 0, bool sync = true) const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    static PytorchDeviceBuffer create(uint64_t size, const PytorchDeviceBuffer* context = nullptr);

    nanovdb::HostBuffer moveToHostBuffer() {
        nanovdb::HostBuffer ret;
        ret.init(mSize, mCpuData);
        mCpuData = nullptr;
        clear();
        return ret;
    }

}; // PytorchDeviceBuffer class

// --------------------------> Implementations below <------------------------------------

inline PytorchDeviceBuffer PytorchDeviceBuffer::create(uint64_t size, const PytorchDeviceBuffer*)
{
    return PytorchDeviceBuffer(size);
}

inline void PytorchDeviceBuffer::init(uint64_t size, void* data)
{
    if (size == mSize)
        return;
    if (mSize > 0)
        this->clear();
    if (size == 0)
        return;
    mSize = size;
    if (data) {
        mCpuData = (uint8_t*) data;
    } else {
        cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
    }
    checkPtr(mCpuData, "failed to allocate host data");
} // PytorchDeviceBuffer::init

inline void PytorchDeviceBuffer::deviceUpload(void* stream, bool sync)
{
    checkPtr(mCpuData, "uninitialized cpu data");
    if (mGpuData == nullptr) {
        mGpuData = reinterpret_cast<uint8_t*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(mSize, reinterpret_cast<cudaStream_t>(stream)));
        // cudaCheck(cudaMalloc((void**)&mGpuData, mSize)); // un-managed memory on the device, always 32B aligned!
    }
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mGpuData, mCpuData, mSize, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    if (sync)
        cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // PytorchDeviceBuffer::gpuUpload

inline void PytorchDeviceBuffer::deviceDownload(void* stream, bool sync) const
{
    checkPtr(mCpuData, "uninitialized cpu data");
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData, mSize, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    if (sync)
        cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // PytorchDeviceBuffer::gpuDownload

inline void PytorchDeviceBuffer::clear()
{
    if (mGpuData)
        c10::cuda::CUDACachingAllocator::raw_delete(mGpuData);
    if (mCpuData)
        cudaCheck(cudaFreeHost(mCpuData));
    mCpuData = mGpuData = nullptr;
    mSize = 0;
} // PytorchDeviceBuffer::clear


template <typename SrcBuf, typename DstBuf>
nanovdb::GridHandle<DstBuf> move(nanovdb::GridHandle<SrcBuf>& buf) {
    return buf;
}


template <>
inline nanovdb::GridHandle<nanovdb::HostBuffer> move(nanovdb::GridHandle<PytorchDeviceBuffer>& buf) {
    return nanovdb::GridHandle<nanovdb::HostBuffer>(buf.buffer().moveToHostBuffer());
}

template <>
inline nanovdb::GridHandle<PytorchDeviceBuffer> move(nanovdb::GridHandle<nanovdb::HostBuffer>& hdl) {
    return nanovdb::GridHandle<PytorchDeviceBuffer>(PytorchDeviceBuffer(hdl.buffer().size(), hdl.buffer().data()));
}


} // namespace fvdb


namespace nanovdb {
    template<>
    struct BufferTraits<fvdb::PytorchDeviceBuffer>
    {
        static const bool hasDeviceDual = true;
    };
} // namespace nanovdb
