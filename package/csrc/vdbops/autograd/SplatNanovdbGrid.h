#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "Kernels.h"
#include "utils/Utils.h"


namespace fvdb {

template <typename BufferType>
struct SplatNanovdbGrid : public torch::autograd::Function<SplatNanovdbGrid<BufferType>> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const VoxelCoordTransform& transform,
                                 Variable points, Variable pointData,
                                 bool returnCounts = false) {

        // Save data for backward in context
        ctx->save_for_backward({pointData, points});

        // NOTE: Passing stuff to backward is ugly and unsafe.
        // TODO: Think about whether grid or transform can get deleted between forward and backward
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;

        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;

        int64_t numOutputValues = grid.template grid<nanovdb::ValueIndex>()->tree().activeVoxelCount();

        auto opts = torch::TensorOptions().dtype(points.dtype()).device(points.device());
        torch::Tensor outGridOrVoxelData = torch::zeros(getShapeButReplaceFirstDim(numOutputValues, pointData), opts);

        torch::Tensor pointDataReshape = pointData.view({pointData.size(0), -1});
        torch::Tensor outGridOrVoxelDataReshape = outGridOrVoxelData.view({outGridOrVoxelData.size(0), -1});

        if (returnCounts) {
            torch::Tensor outCounts = torch::zeros({numOutputValues}, opts);
            dispatchSplatIntoGridTrilinearWithCounts(
                grid, points, pointDataReshape, transform, outGridOrVoxelDataReshape, outCounts);
            return {outGridOrVoxelData, outCounts};
        } else {
            dispatchSplatIntoGridTrilinear(
                grid, points, pointDataReshape, transform, outGridOrVoxelDataReshape);
            return {outGridOrVoxelData};
        }
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        // Use data saved in forward
        variable_list saved = ctx->get_saved_variables();
        Variable pointData = saved.at(0);
        Variable points = saved.at(1);
        const VoxelCoordTransform& transform = *((VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());

        torch::Tensor pointDataReshape = pointData.reshape({pointData.size(0), -1});

        // FIXME handle more dimensions
        Variable gradOut = grad_output.at(0);  // [#voxels | #corners, #feats]
        Variable outGradIn = torch::zeros_like(pointDataReshape);  // [#points, #feats]

        dispatchSampleGridTrilinear<BufferType>(grid, points, gradOut, transform, outGradIn);
        // dispatchSampleGridGrad<BufferType>(grid, points, gradOut, transform, outGrad);

        std::vector<long> outShape(pointData.dim());
        outShape[0] = outGradIn.size(0);
        for (unsigned long i = 1; i < outShape.size(); i += 1) {
            outShape[i] = pointData.size(i);
        }
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), outGradIn.reshape(outShape), torch::Tensor()};
    }
};

} // namespace fvdb