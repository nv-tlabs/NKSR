#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "Kernels.h"
#include "utils/Utils.h"



namespace fvdb {

template <typename BufferType>
struct UpsampleNanovdbGrid : public torch::autograd::Function<UpsampleNanovdbGrid<BufferType>> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& coarseGrid,
                                 const nanovdb::GridHandle<BufferType>& fineGrid,
                                 unsigned upsamplingFactor,
                                 Variable coarseData) {

        TORCH_CHECK(coarseData.dim() > 1,
                    "coarse_data must have more than one dimension. i.e. have shape (num_voxels, *)");
        TORCH_CHECK(coarseData.size(0) == (int64_t) coarseGrid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount(),
                    "coarse_data must have the same number of voxels as coarse_grid");

        // Save data for backward in context
        ctx->save_for_backward({coarseData});

        // NOTE: Passing stuff to backward is ugly and unsafe.
        // TODO: Think about whether grid or transform can get deleted between forward and backward
        auto coarseGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        coarseGridPtr->ShareExternal((void*) &coarseGrid, caffe2::TypeMeta());
        ctx->saved_data["coarse_grid"] = coarseGridPtr;

        auto fineGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        fineGridPtr->ShareExternal((void*) &fineGrid, caffe2::TypeMeta());
        ctx->saved_data["fine_grid"] = fineGridPtr;

        ctx->saved_data["upsampling_factor"] = (int64_t) upsamplingFactor;

        int64_t numOutputValues = fineGrid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount();

        auto opts = torch::TensorOptions().dtype(coarseData.dtype()).device(coarseData.device());
        torch::Tensor outFineData = torch::zeros(getShapeButReplaceFirstDim(numOutputValues, coarseData), opts);

        torch::Tensor coarseDataReshape = coarseData.view({coarseData.size(0), -1});
        torch::Tensor outFineDataReshape = outFineData.view({outFineData.size(0), -1});

        dispatchUpsampleGridNearest<BufferType>(
            coarseGrid, fineGrid,
            coarseDataReshape, outFineDataReshape, upsamplingFactor);

        return {outFineData};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        // // Use data saved in forward
        variable_list saved = ctx->get_saved_variables();
        Variable coarseData = saved.at(0);

        const nanovdb::GridHandle<BufferType>& coarseGrid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["coarse_grid"].toBlob()->GetRaw());
        const nanovdb::GridHandle<BufferType>& fineGrid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["fine_grid"].toBlob()->GetRaw());
        const int64_t upsamplingFactor = ctx->saved_data["upsampling_factor"].toInt();

        torch::Tensor coarseDataReshape = coarseData.view({coarseData.size(0), -1});

        Variable gradOut = grad_output.at(0);  // [#fine_voxels | #fine_corners, *]
        Variable gradOutReshape = gradOut.view({gradOut.size(0), -1});
        Variable outGradInReshape = torch::zeros_like(coarseDataReshape);  // [#coarse_voxels | #coarse_corners, *]

        dispatchUpsampleGridNearestGrad<BufferType>(
            fineGrid, coarseGrid,
            gradOutReshape /* fineData */,
            outGradInReshape /* outCoarseData */,
            upsamplingFactor
        );

        Variable outGradIn = outGradInReshape.reshape(getShapeButReplaceFirstDim(coarseData.size(0), gradOut));
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), outGradIn};
    }
};

} // namespace fvdb