#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "Kernels.h"
#include "utils/Utils.h"


namespace fvdb {

enum SampleMode {
    TRILINEAR, BEZIER
};

template <typename BufferType>
struct SampleNanovdbGrid : public torch::autograd::Function<SampleNanovdbGrid<BufferType>> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const VoxelCoordTransform& transform,
                                 Variable points, Variable data,
                                 bool returnGrad = false,
                                 SampleMode mode = TRILINEAR) {

        // Save data for backward in context
        ctx->save_for_backward({data, points});

        // NOTE: Passing stuff to backward is ugly and unsafe.
        // TODO: Think about whether grid or transform can get deleted between forward and backward
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;

        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;
        ctx->saved_data["return_grad"] = returnGrad;
        ctx->saved_data["mode"] = (int) mode;

        auto opts = torch::TensorOptions().dtype(data.dtype())
                                          .device(data.device())
                                          .requires_grad(data.requires_grad());
        torch::Tensor dataReshape = data.reshape({data.size(0), -1});
        torch::Tensor outFeatures = torch::zeros({points.size(0), dataReshape.size(1)}, opts);

        std::vector<long> outShape(data.dim());
        outShape[0] = outFeatures.size(0);
        for (unsigned long i = 1; i < outShape.size(); i += 1) {
            outShape[i] = data.size(i);
        }

        if (returnGrad) {
            torch::Tensor pointsGrad = torch::zeros({points.size(0), dataReshape.size(1), 3}, opts);
            if (mode == BEZIER) {
                dispatchSampleGridBezierWithGrad(grid, points, dataReshape, transform, outFeatures, pointsGrad);
            } else {
                dispatchSampleGridTrilinearWithGrad(grid, points, dataReshape, transform, outFeatures, pointsGrad);
            }

            std::vector<long> gradOutShape = outShape; gradOutShape.push_back(3);
            return {outFeatures.reshape(outShape), pointsGrad.reshape(gradOutShape)};
        } else {
            if (mode == BEZIER) {
                dispatchSampleGridBezier(grid, points, dataReshape, transform, outFeatures);
            } else {
                // default
                dispatchSampleGridTrilinear(grid, points, dataReshape, transform, outFeatures);
            }
            return {outFeatures.reshape(outShape)};
        }
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        // Use data saved in forward
        variable_list saved = ctx->get_saved_variables();
        Variable data = saved.at(0);
        Variable points = saved.at(1);
        const VoxelCoordTransform& transform = *((VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());
        bool returnGrad = ctx->saved_data["return_grad"].toBool();
        SampleMode mode = (SampleMode) ctx->saved_data["mode"].toInt();

        torch::Tensor dataReshape = data.reshape({data.size(0), -1});

        // FIXME handle more dimensions
        Variable gradOut = grad_output.at(0);  // [#points, #feats]
        Variable outGrad = torch::zeros_like(dataReshape);  // [#voxels, #feats]

        std::vector<long> outShape(data.dim());
        outShape[0] = outGrad.size(0);
        for (unsigned long i = 1; i < outShape.size(); i += 1) {
            outShape[i] = data.size(i);
        }

        if (returnGrad) {
            Variable gradPtsGrad = grad_output.at(1);  // [#points, #feats, 3]
            if (mode == BEZIER) {
                dispatchSampleGridBezierWithGradGrad(grid, points, gradOut, gradPtsGrad, transform, outGrad);
            } else {
                dispatchSampleGridTrilinearWithGradGrad(grid, points, gradOut, gradPtsGrad, transform, outGrad);
            }
        } else {
            if (mode == BEZIER) {
                dispatchSplatIntoGridBezier<BufferType>(grid, points, gradOut, transform, outGrad);
            } else {
                dispatchSplatIntoGridTrilinear<BufferType>(grid, points, gradOut, transform, outGrad);
            }
        }

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), outGrad.reshape(outShape), torch::Tensor(), torch::Tensor()};
    }
};

} // namespace fvdb