#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "Kernels.h"
#include "utils/Utils.h"


namespace fvdb {

template <typename BufferType>
struct MaxPoolNanovdbGrid : public torch::autograd::Function<MaxPoolNanovdbGrid<BufferType>> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& fineGrid,
                                 const nanovdb::GridHandle<BufferType>& coarseGrid,
                                 unsigned poolingFactor,
                                 Variable fineData) {

        TORCH_CHECK(fineData.dim() > 1,
                    "fine_data must have more than one dimension. i.e. have shape (num_voxels, *)");
        TORCH_CHECK(fineData.size(0) == (int64_t) fineGrid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount(),
                    "fine_data must have the same number of voxels as fine_grid");

        ctx->save_for_backward({fineData});

        auto fineGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        fineGridPtr->ShareExternal((void*) &fineGrid, caffe2::TypeMeta());
        ctx->saved_data["fine_grid"] = fineGridPtr;

        auto coarseGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        coarseGridPtr->ShareExternal((void*) &coarseGrid, caffe2::TypeMeta());
        ctx->saved_data["coarse_grid"] = coarseGridPtr;

        ctx->saved_data["pooling_factor"] = (int64_t) poolingFactor;

        // std::cout << "here" << std::endl << std::flush;
        int64_t numOutputValues = coarseGrid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount();
        auto opts = torch::TensorOptions().dtype(fineData.dtype()).device(fineData.device());
        torch::Tensor outCoarseData = torch::empty(getShapeButReplaceFirstDim(numOutputValues, fineData), opts);

        // std::cout << "there" << std::endl << std::flush;
        torch::Tensor fineDataReshape = fineData.view({fineData.size(0), -1});
        // std::cout << "everywhere" << std::endl << std::flush;
        torch::Tensor outCoarseDataReshape = outCoarseData.view({outCoarseData.size(0), -1});
        // std::cout << "nowhere" << std::endl << std::flush;
        dispatchDownsampleGridMaxPool<BufferType>(
            fineGrid, coarseGrid,
            fineDataReshape, outCoarseDataReshape, poolingFactor);

        return {outCoarseData};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        // Use data saved in forward
        variable_list saved = ctx->get_saved_variables();
        Variable fineData = saved.at(0);
        const nanovdb::GridHandle<BufferType>& fineGrid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["fine_grid"].toBlob()->GetRaw());
        const nanovdb::GridHandle<BufferType>& coarseGrid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["coarse_grid"].toBlob()->GetRaw());
        const int64_t poolingFactor = ctx->saved_data["pooling_factor"].toInt();

        torch::Tensor fineDataReshape = fineData.view({fineData.size(0), -1});

        Variable gradOut = grad_output.at(0);  // [#coarse_voxels | #coarse_corners, *]
        Variable gradOutReshape = gradOut.view({gradOut.size(0), -1});
        Variable outGradInReshape = torch::zeros_like(fineDataReshape);  // [#fine_voxels | #fine_corners, *]

        dispatchDownsampleGridMaxPoolGrad<BufferType>(
            coarseGrid, fineGrid,
            fineData,
            gradOutReshape,
            outGradInReshape,
            poolingFactor
        );

        Variable outGradIn = outGradInReshape.reshape(getShapeButReplaceFirstDim(fineData.size(0), gradOut));
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), outGradIn};
    }
};

} // namespace fvdb