
#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "PytorchDeviceBuffer.h"

#include <c10/cuda/CUDAGuard.h>

namespace fvdb {

template <typename BufferType>
inline void maybeUploadToGPU(nanovdb::GridHandle<BufferType>& hdl, c10::DeviceIndex deviceIndex, bool sync = false) {}

template <>
inline void maybeUploadToGPU<PytorchDeviceBuffer>(nanovdb::GridHandle<PytorchDeviceBuffer>& hdl, c10::DeviceIndex deviceIndex, bool sync) {
    at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(deviceIndex);
    hdl.deviceUpload(defaultStream.stream(), sync);
}

template <typename BufferType>
inline bool isCuda() { return false; }

template <>
inline bool isCuda<PytorchDeviceBuffer>() { return true; }



template <typename BufferType>
inline void checkDevice(const torch::Tensor& t, c10::DeviceIndex deviceIndex);

template <>
inline void checkDevice<nanovdb::HostBuffer>(const torch::Tensor& t, c10::DeviceIndex deviceIndex) {
    TORCH_CHECK(t.is_cpu(), "All tensors must be on the CPU");
}

template <>
inline void checkDevice<PytorchDeviceBuffer>(const torch::Tensor& t, c10::DeviceIndex deviceIndex) {
    auto thisDev = torch::Device(torch::DeviceType::CUDA, deviceIndex);
    TORCH_CHECK(t.is_cuda(), "All tensors must be on the GPU");
    TORCH_CHECK(t.device() == thisDev, "All tensors must be on the same device (" +
                                       thisDev.str() +
                                       ") as index grid, but got " + t.device().str());
}

template <typename BufferT>
inline constexpr c10::DeviceIndex defaultDeviceIndex() {
    return 0;
}

template <>
inline constexpr c10::DeviceIndex defaultDeviceIndex<nanovdb::HostBuffer>() {
    return -1;
}


template <typename BufferType>
inline torch::Device bufferToTorchDevice(c10::DeviceIndex index);

template <>
inline torch::Device bufferToTorchDevice<nanovdb::HostBuffer>(c10::DeviceIndex index) {
    return torch::Device(torch::kCPU, -1);
}

template <>
inline torch::Device bufferToTorchDevice<PytorchDeviceBuffer>(c10::DeviceIndex index) {
    return torch::Device(torch::kCUDA, index);
}

}