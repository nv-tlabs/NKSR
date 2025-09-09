#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "Kernels.h"
#include "utils/Utils.h"


namespace fvdb {


template <typename BufferType>
struct TransformPoints : public torch::autograd::Function<TransformPoints<BufferType>> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const VoxelCoordTransform& transform,
                                 Variable points,
                                 bool isInverse) {

        // ctx->saved_data["fine_data"] = fineData;

        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;
        ctx->saved_data["isInverse"] = isInverse;

        auto opts = torch::TensorOptions().dtype(points.dtype()).device(points.device());
        torch::Tensor outTxPoints = torch::empty({points.size(0), points.size(1)}, opts);

        if (isInverse) {
            dispatchInvTransformPointsToGrid<BufferType>(
                points, transform, outTxPoints);
        } else {
            dispatchTransformPointsToGrid<BufferType>(
                points, transform, outTxPoints);
        }

        return {outTxPoints};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        Variable gradOut = grad_output.at(0);

        // Use data saved in forward
        const VoxelCoordTransform& transform = *((VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());

        Variable outGradIn = torch::empty_like(gradOut);  // [#fine_voxels | #fine_corners, *]

        const bool isInverse = ctx->saved_data["isInverse"].toBool();

        if (isInverse) {
            dispatchInvTransformPointsToGridGrad<BufferType>(
                gradOut, transform, outGradIn);

        } else {
            dispatchTransformPointsToGridGrad<BufferType>(
                gradOut, transform, outGradIn);
        }

        // Variable outGradIn = outGradInReshape.reshape(getShapeButReplaceFirstDim(fineData.size(0), gradOut));
        return {torch::Tensor(), outGradIn, torch::Tensor()};
    }
};


} // namespace fvdb