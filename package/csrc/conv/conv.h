#include <torch/extension.h>
#include <torch/autograd.h>
#include <SparseFeatureIndexGrid.h>

template <typename BufferType>
torch::Tensor convolutionKernelMap(const SparseFeatureIndexGrid<BufferType> &source,
                                   const SparseFeatureIndexGrid<BufferType> &target,
                                   int kernel,
                                   unsigned nThreadsX = 128);

void convolution_forward_cpu(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor kernel, at::Tensor neighbor_map,
                             at::Tensor neighbor_offset, const bool transpose);

void convolution_backward_cpu(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor kernel,
                              at::Tensor grad_kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset,
                              const bool transpose);

void convolution_forward_cuda(at::Tensor in_feat, at::Tensor out_feat,
                              at::Tensor kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset,
                              const bool transpose);

void convolution_backward_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);

using variable_list = torch::autograd::variable_list;
using AutogradContext = torch::autograd::AutogradContext;
using Variable = torch::autograd::Variable;

struct SparseConvolution : public torch::autograd::Function<SparseConvolution> {
    static variable_list forward(AutogradContext *ctx,
                                 Variable input, Variable weight,
                                 Variable nbmaps, Variable nbsizes,
                                 std::vector<int> sizes, bool transposed) {
        torch::Tensor output;
        auto opt = torch::TensorOptions().dtype(input.dtype()).device(input.device());
        if (!transposed) {
            output = torch::zeros({sizes[1], weight.size(-1)}, opt);
        } else {
            output = torch::zeros({sizes[0], weight.size(-1)}, opt);
        }

        if (input.is_cuda()) {
            convolution_forward_cuda(input, output, weight, nbmaps, nbsizes.cpu(), transposed);
        } else {
            convolution_forward_cpu(input, output, weight, nbmaps, nbsizes.cpu(), transposed);
        }

        // Save for backward
        ctx->saved_data["input"] = input;
        ctx->saved_data["weight"] = weight;
        ctx->saved_data["nbmaps"] = nbmaps;
        ctx->saved_data["nbsizes"] = nbsizes;
        ctx->saved_data["transposed"] = transposed;

        return {output};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        Variable input = ctx->saved_data["input"].toTensor();
        Variable weight = ctx->saved_data["weight"].toTensor();
        Variable nbmaps = ctx->saved_data["nbmaps"].toTensor();
        Variable nbsizes = ctx->saved_data["nbsizes"].toTensor();
        bool transposed = ctx->saved_data["transposed"].toBool();

        torch::Tensor gradInput = torch::zeros_like(input);
        torch::Tensor gradWeight = torch::zeros_like(weight);

        Variable gradOut = grad_output.at(0);
        if (gradOut.is_cuda()) {
            convolution_backward_cuda(input, gradInput, gradOut.contiguous(), weight, gradWeight, nbmaps, nbsizes.cpu(), transposed);
        } else {
            convolution_backward_cpu(input, gradInput, gradOut.contiguous(), weight, gradWeight, nbmaps, nbsizes.cpu(), transposed);
        }

        return {gradInput, gradWeight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};
