#include <torch/extension.h>
#include "conv.h"

template <typename GridT>
void makeBinding(pybind11::module_& m) {
    m.def("convolution_kernel_map", [](std::shared_ptr<GridT> source, std::shared_ptr<GridT> target, int kernel) -> torch::Tensor {
        return convolutionKernelMap<typename GridT::BufferT>(*source, *target, kernel);
    });
}

void pybind_conv(py::module& m) {
    makeBinding<SparseFeatureIndexGrid<nanovdb::HostBuffer>>(m);
    makeBinding<SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>>(m);

    m.def("sparse_convolution", [](torch::Tensor input, torch::Tensor weight,
                                   torch::Tensor nbmaps, torch::Tensor nbsizes,
                                   const std::vector<int>& sizes, bool transposed) {
        TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
        TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
        TORCH_CHECK(nbmaps.is_contiguous() && nbmaps.scalar_type() == torch::kInt32, "nbmaps must be contiguous");
        TORCH_CHECK(nbsizes.is_contiguous() && nbsizes.scalar_type() == torch::kInt32, "nbsizes must be contiguous");
        return SparseConvolution::apply(input, weight, nbmaps, nbsizes, sizes, transposed)[0];
    }, "Perform Sparse Convolution.");
}
