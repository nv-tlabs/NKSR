#include "../common/platform.h"
//#include "../common/math_util.h"

#include <utils/VoxelCoordTransform.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/extension.h>
#include <torch/autograd.h>
#include <SparseFeatureIndexGrid.h>


using variable_list = torch::autograd::variable_list;
using AutogradContext = torch::autograd::AutogradContext;
using Variable = torch::autograd::Variable;

// For active nodes iteration, need some subdivision to divide computation
constexpr int AC_NUM_BLOCK = fvdb::IndexTree::LeafNodeType::NUM_VALUES;
constexpr int AC_EACH_CHUNK = 4;
constexpr int AC_NUM_CHUNKS = AC_NUM_BLOCK / AC_EACH_CHUNK;

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline ScalarT bezier_1dim(const ScalarT x) {
    static const ScalarT OPF = 1.5;
    static const ScalarT PF = 0.5;

    bool r1 = x < -OPF, r2 = x < -PF, r3 = x < PF, r4 = x < OPF;
    if (!r1 && r2) {
        return (x + OPF) * (x + OPF);
    } else if (!r2 && r3) {
        return -(ScalarT) 2 * x * x + OPF;
    } else if (!r3 && r4) {
        return (x - OPF) * (x - OPF);
    }
    return 0.0;
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline ScalarT bezier_grad_1dim(const ScalarT x) {
    bool r1 = x < -1.5, r2 = x < -0.5, r3 = x < 0.5, r4 = x < 1.5;
    if (!r1 && r2) {
        return 2 * x + 3;
    } else if (!r2 && r3) {
        return -4 * x;
    } else if (!r3 && r4) {
        return 2 * x - 3;
    }
    return 0.0;
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline bool has_overlap(
        const nanovdb::Vec3<ScalarT>& a_min, const nanovdb::Vec3<ScalarT>& a_max,
        const nanovdb::Vec3<ScalarT>& b_min, const nanovdb::Vec3<ScalarT>& b_max) {
    return (a_max[0] >= b_min[0] && b_max[0] >= a_min[0]) &&
           (a_max[1] >= b_min[1] && b_max[1] >= a_min[1]) &&
           (a_max[2] >= b_min[2] && b_max[2] >= a_min[2]);
}

template <typename ScalarT, typename Acc2, typename Acc3>
_CPU_AND_GPU_CODE_ inline void kernel_grad_evaluation_fwd(
        const int64_t offset, const int64_t pi, ScalarT scale,
        const ScalarT& pcx, const ScalarT& pcy, const ScalarT& pcz,
        const Acc2& pKernel,
        const Acc2& cKernel,
        const Acc3& gradKernelQuery,
        bool grad,
        // Output (necessary)
        ScalarT& f, nanovdb::Vec3<ScalarT>& gradF,
        // Output (for backward)
        ScalarT& bezierKernel, ScalarT& dpKernel, nanovdb::Vec3<ScalarT>& db) {

    // Kernel evaluation
    ScalarT bx = bezier_1dim(pcx), by = bezier_1dim(pcy), bz = bezier_1dim(pcz);
    bezierKernel = bx * by * bz;

    dpKernel = static_cast<ScalarT>(1.0);
    // For the Dot-Product part it could be a trivial one.
    if (cKernel.size(1) > 0) {
        dpKernel = static_cast<ScalarT>(0.0);
        for (int64_t k = 0; k < cKernel.size(1); ++k) {
//            dpKernel = fmaf(cKernel[offset][k], pKernel[pi][k], dpKernel);
            dpKernel += cKernel[offset][k] * pKernel[pi][k];
        }
    }
    f = bezierKernel * dpKernel;

    // Gradient Kernel evaluation
    if (!grad) return;

    ScalarT dbx = bezier_grad_1dim(pcx) * scale;
    ScalarT dby = bezier_grad_1dim(pcy) * scale;
    ScalarT dbz = bezier_grad_1dim(pcz) * scale;
    db[0] = dbx * by * bz;
    db[1] = bx * dby * bz;
    db[2] = bx * by * dbz;

#pragma unroll
    for (int dim = 0; dim < 3; ++ dim) {
        auto gradDot = static_cast<ScalarT>(0.0);
        if (cKernel.size(1) > 0 && gradKernelQuery.size(0) > 0) {
            for (int64_t k = 0; k < cKernel.size(1); ++k) {
                gradDot += cKernel[offset][k] * gradKernelQuery[pi][k][dim];
            }
        }
        gradF[dim] = bezierKernel * gradDot + dpKernel * db[dim];
    }
}

template <typename ScalarT, bool isCompound, typename Acc1, typename Acc2, typename Acc3>
_CPU_AND_GPU_CODE_ inline void kernel_grad_evaluation_bwd(
        const int64_t offset, const int64_t pi,
        const Acc2& pKernel, const Acc2& cKernel,
        const Acc3& gradKernelQuery,
        // Upstream gradient input
        const Acc1& gradOutFunc,
        const Acc2& gradOutGradFunc,    // not used if chaining
        bool grad, const ScalarT alpha,
        // Downstream gradient output
        Acc2& gradQueryKernel, Acc2& gradGridKernel,
        Acc1& gradGridAlpha, Acc3& gradGradKernelQuery,
        // Compound input
        int64_t outIdx,
        const ScalarT of, const nanovdb::Vec3<ScalarT>& gradOF,
        // My forward output
        const ScalarT f, const nanovdb::Vec3<ScalarT>& gradF,
        const ScalarT& bezierKernel, const ScalarT& dpKernel, const nanovdb::Vec3<ScalarT>& db
) {

    if (!grad) {
        ScalarT gradUpper;
        if (!isCompound) { gradUpper = gradOutFunc[pi] * alpha; }
        else {
            if (outIdx == -1) {
                gradUpper = of;
            } else {
                gradUpper = gradOutFunc[outIdx] * of;
            }
        }

        if (cKernel.size(1) > 0) {
            for (int64_t k = 0; k < cKernel.size(1); ++k) {
                ScalarT dfunc = gradUpper * bezierKernel;
                atomicAdd(&gradQueryKernel[pi][k], dfunc * cKernel[offset][k]);
                atomicAdd(&gradGridKernel[offset][k], dfunc * pKernel[pi][k]);
            }
        }
        if (!isCompound) {
            atomicAdd(&gradGridAlpha[offset], f * gradOutFunc[pi]);
        }
    } else {

#pragma unroll
        for (int dim = 0; dim < 3; ++ dim) {
            ScalarT dalpha;
            if (!isCompound) { dalpha = gradOutGradFunc[pi][dim] * alpha; }
            else {
                if (outIdx == -1) {
                    dalpha = gradOF[dim];
                } else {
                    dalpha = gradOutFunc[outIdx] * gradOF[dim];
                }
            }

            if (cKernel.size(1) > 0) {
                for (int64_t k = 0; k < cKernel.size(1); ++k) {
                    atomicAdd(&gradQueryKernel[pi][k], dalpha * cKernel[offset][k] * db[dim]);
                    atomicAdd(&gradGridKernel[offset][k], dalpha * pKernel[pi][k] * db[dim]);

                    if (gradKernelQuery.size(0) > 0) {
                        atomicAdd(&gradGridKernel[offset][k], dalpha * bezierKernel * gradKernelQuery[pi][k][dim]);
                        atomicAdd(&gradGradKernelQuery[pi][k][dim], dalpha * bezierKernel * cKernel[offset][k]);
                    }
                }
            }

            if (!isCompound) {
                atomicAdd(&gradGridAlpha[offset], gradF[dim] * gradOutGradFunc[pi][dim]);
            }
        }
    }

}


template <typename BufferType>
void dispatchKernelEvaluation(const nanovdb::GridHandle<BufferType>& grid,
                              const fvdb::VoxelCoordTransform& transform,
                              const torch::Tensor& query,
                              const torch::Tensor& queryKernel,
                              const torch::Tensor& gridKernel,
                              const torch::Tensor& gridAlpha,
                              const torch::Tensor& gradKernelQuery,
                              torch::Tensor& outFunc,
                              torch::Tensor& outGradFunc,
                              unsigned nThreadsX = 128);

template <typename BufferType>
void dispatchKernelEvaluationBackward(const nanovdb::GridHandle<BufferType>& grid,
                                      const fvdb::VoxelCoordTransform& transform,
                                      const torch::Tensor& query,
                                      const torch::Tensor& queryKernel,
                                      const torch::Tensor& gridKernel,
                                      const torch::Tensor& gridAlpha,
                                      const torch::Tensor& gradKernelQuery,
                                      const torch::Tensor& gradOutFunc,
                                      const torch::Tensor& gradOutGradFunc,
                                      torch::Tensor& gradQueryKernel,
                                      torch::Tensor& gradGridKernel,
                                      torch::Tensor& gradGridAlpha,
                                      torch::Tensor& gradGradKernelQuery,
                                      unsigned nThreadsX = 128);

template <typename BufferType>
struct KernelEvaluation : public torch::autograd::Function<KernelEvaluation<BufferType>> {
    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const fvdb::VoxelCoordTransform& transform,
                                 bool grad,
                                 Variable query, Variable queryKernel,
                                 Variable gridKernel, Variable gridAlpha,
                                 Variable gradKernelQuery) {
        // Save for backward
        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;
        ctx->saved_data["query"] = query;
        ctx->saved_data["queryKernel"] = queryKernel;
        ctx->saved_data["gridKernel"] = gridKernel;
        ctx->saved_data["gridAlpha"] = gridAlpha;
        ctx->saved_data["gradKernelQuery"] = gradKernelQuery;

        // Prepare output
        auto opts = torch::TensorOptions().dtype(queryKernel.dtype())
                .device(queryKernel.device());
        torch::Tensor outFunc = torch::zeros(query.size(0), opts);
        torch::Tensor outGradFunc = torch::zeros({0, 3}, opts);
        if (grad) {
            outGradFunc = torch::zeros({query.size(0), 3}, opts);
        }

        dispatchKernelEvaluation(grid, transform, query, queryKernel, gridKernel, gridAlpha, gradKernelQuery, outFunc, outGradFunc);
        return {outFunc, outGradFunc};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& transform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        Variable query = ctx->saved_data["query"].toTensor();
        Variable queryKernel = ctx->saved_data["queryKernel"].toTensor();
        Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
        Variable gridAlpha = ctx->saved_data["gridAlpha"].toTensor();
        Variable gradKernelQuery = ctx->saved_data["gradKernelQuery"].toTensor();

        // Prepare grad input
        Variable gradOutFunc = grad_output.at(0);
        Variable gradOutGradFunc = grad_output.at(1);

        // Prepare output
        torch::Tensor gradQueryKernel = torch::zeros_like(queryKernel);
        torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
        torch::Tensor gradGridAlpha = torch::zeros_like(gridAlpha);
        torch::Tensor gradGradKernelQuery = torch::zeros_like(gradKernelQuery);

        dispatchKernelEvaluationBackward<BufferType>(grid, transform, query, queryKernel, gridKernel, gridAlpha,
                                                     gradKernelQuery, gradOutFunc, gradOutGradFunc,
                                                     gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery);
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery};
    }
};

template <typename BufferType>
torch::Tensor buildCOOIndexer(const SparseFeatureIndexGrid<BufferType>& iSVH,
                              const SparseFeatureIndexGrid<BufferType>& jSVH,
                              unsigned nThreadsX = 128);

template <typename BufferType>
void dispatchMatrixBuilding(const nanovdb::GridHandle<BufferType>& gridI,
                            const nanovdb::GridHandle<BufferType>& gridJ,
                            const fvdb::VoxelCoordTransform& transformI,
                            const fvdb::VoxelCoordTransform& transformJ,
                            const torch::Tensor& ptsPos,
                            const torch::Tensor& ptsKernelI,
                            const torch::Tensor& ptsKernelJ,
                            const torch::Tensor& iKernel,
                            const torch::Tensor& jKernel,
                            const torch::Tensor& gradPtsKernelPosI,
                            const torch::Tensor& gradPtsKernelPosJ,
                            const torch::Tensor& indexMap,
                            bool grad,
                            torch::Tensor& outMatrix,
                            unsigned nThreadsX = 32,
                            unsigned nThreadsY = 10);

template <typename BufferType>
void dispatchMatrixBuildingBackward(const nanovdb::GridHandle<BufferType>& gridI,
                                    const nanovdb::GridHandle<BufferType>& gridJ,
                                    const fvdb::VoxelCoordTransform& transformI,
                                    const fvdb::VoxelCoordTransform& transformJ,
                                    const torch::Tensor& ptsPos,
                                    const torch::Tensor& ptsKernelI,
                                    const torch::Tensor& ptsKernelJ,
                                    const torch::Tensor& iKernel,
                                    const torch::Tensor& jKernel,
                                    const torch::Tensor& gradPtsKernelPosI,
                                    const torch::Tensor& gradPtsKernelPosJ,
                                    const torch::Tensor& indexMap,
                                    bool grad,
                                    const torch::Tensor& gradOutMatrix,
                                    torch::Tensor& gradPtsKernelI,
                                    torch::Tensor& gradPtsKernelJ,
                                    torch::Tensor& gradIKernel,
                                    torch::Tensor& gradJKernel,
                                    torch::Tensor& gradGradPtsKernelPosI,
                                    torch::Tensor& gradGradPtsKernelPosJ,
                                    unsigned nThreadsX = 16,
                                    unsigned nThreadsY = 10);

template <typename BufferType>
struct MatrixBuilding : public torch::autograd::Function<MatrixBuilding<BufferType>> {
    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType> &iGrid,
                                 const nanovdb::GridHandle<BufferType> &jGrid,
                                 const fvdb::VoxelCoordTransform &iTransform,
                                 const fvdb::VoxelCoordTransform &jTransform,
                                 Variable ptsPos,
                                 Variable ptsKernelI, Variable ptsKernelJ,
                                 Variable iKernel, Variable jKernel,
                                 Variable gradPtsKernelPosI, Variable gradPtsKernelPosJ,
                                 Variable indexMap,
                                 bool grad, int64_t numEntries) {
        // Save for backward
        auto iGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        iGridPtr->ShareExternal((void *) &iGrid, caffe2::TypeMeta());
        ctx->saved_data["iGrid"] = iGridPtr;
        auto jGridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        jGridPtr->ShareExternal((void *) &jGrid, caffe2::TypeMeta());
        ctx->saved_data["jGrid"] = jGridPtr;
        auto iTransformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        iTransformPtr->ShareExternal((void *) &iTransform, caffe2::TypeMeta());
        ctx->saved_data["iTransform"] = iTransformPtr;
        auto jTransformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        jTransformPtr->ShareExternal((void *) &jTransform, caffe2::TypeMeta());
        ctx->saved_data["jTransform"] = jTransformPtr;
        ctx->saved_data["ptsPos"] = ptsPos;
        ctx->saved_data["ptsKernelI"] = ptsKernelI;
        ctx->saved_data["ptsKernelJ"] = ptsKernelJ;
        ctx->saved_data["iKernel"] = iKernel;
        ctx->saved_data["jKernel"] = jKernel;
        ctx->saved_data["gradPtsKernelPosI"] = gradPtsKernelPosI;
        ctx->saved_data["gradPtsKernelPosJ"] = gradPtsKernelPosJ;
        ctx->saved_data["indexMap"] = indexMap;
        ctx->saved_data["grad"] = grad;

        // Prepare output
        auto opts = torch::TensorOptions().dtype(ptsKernelI.dtype())
                .device(ptsKernelI.device());
        torch::Tensor outMatrix = torch::zeros(numEntries, opts);

        dispatchMatrixBuilding(
                iGrid, jGrid, iTransform, jTransform,
                ptsPos, ptsKernelI, ptsKernelJ, iKernel, jKernel, gradPtsKernelPosI, gradPtsKernelPosJ,
                indexMap, grad, outMatrix);
        return {outMatrix};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const nanovdb::GridHandle<BufferType>& iGrid =
                *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["iGrid"].toBlob()->GetRaw());
        const nanovdb::GridHandle<BufferType>& jGrid =
                *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["jGrid"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& iTransform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["iTransform"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& jTransform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["jTransform"].toBlob()->GetRaw());
        Variable ptsPos = ctx->saved_data["ptsPos"].toTensor();
        Variable ptsKernelI = ctx->saved_data["ptsKernelI"].toTensor();
        Variable ptsKernelJ = ctx->saved_data["ptsKernelJ"].toTensor();
        Variable iKernel = ctx->saved_data["iKernel"].toTensor();
        Variable jKernel = ctx->saved_data["jKernel"].toTensor();
        Variable gradPtsKernelPosI = ctx->saved_data["gradPtsKernelPosI"].toTensor();
        Variable gradPtsKernelPosJ = ctx->saved_data["gradPtsKernelPosJ"].toTensor();
        Variable indexMap = ctx->saved_data["indexMap"].toTensor();
        bool grad = ctx->saved_data["grad"].toBool();

        // Prepare grad input
        Variable gradOutMatrix = grad_output.at(0);

        // Prepare output
        torch::Tensor gradPtsKernelI = torch::zeros_like(ptsKernelI);
        torch::Tensor gradPtsKernelJ = torch::zeros_like(ptsKernelJ);
        torch::Tensor gradIKernel = torch::zeros_like(iKernel);
        torch::Tensor gradJKernel = torch::zeros_like(jKernel);
        torch::Tensor gradGradPtsKernelPosI = torch::zeros_like(gradPtsKernelPosI);
        torch::Tensor gradGradPtsKernelPosJ = torch::zeros_like(gradPtsKernelPosJ);

        dispatchMatrixBuildingBackward(
                iGrid, jGrid, iTransform, jTransform,
                ptsPos, ptsKernelI, ptsKernelJ, iKernel, jKernel, gradPtsKernelPosI, gradPtsKernelPosJ, indexMap,
                grad, gradOutMatrix, gradPtsKernelI, gradPtsKernelJ, gradIKernel, gradJKernel,
                gradGradPtsKernelPosI, gradGradPtsKernelPosJ);

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(),
                gradPtsKernelI, gradPtsKernelJ, gradIKernel, gradJKernel,
                gradGradPtsKernelPosI, gradGradPtsKernelPosJ,
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

template <typename BufferType>
void dispatchKBuilding(const nanovdb::GridHandle<BufferType>& grid,
                       const fvdb::VoxelCoordTransform& transform,
                       const torch::Tensor& kernel,
                       const torch::Tensor& indexMap,
                       torch::Tensor& outMatrix,
                       unsigned nThreadsX = 128);

template <typename BufferType>
void dispatchKBuildingBackward(const nanovdb::GridHandle<BufferType>& grid,
                               const fvdb::VoxelCoordTransform& transform,
                               const torch::Tensor& kernel,
                               const torch::Tensor& indexMap,
                               const torch::Tensor& gradOutMatrix,
                               torch::Tensor& gradKernel,
                               unsigned nThreadsX = 128);


template <typename BufferType>
struct KBuilding : public torch::autograd::Function<KBuilding<BufferType>> {
    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const fvdb::VoxelCoordTransform& transform,
                                 Variable kernel, Variable indexMap, int64_t numEntries) {

        // Save for backward
        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;
        ctx->saved_data["kernel"] = kernel;
        ctx->saved_data["indexMap"] = indexMap;

        // Prepare output
        auto opts = torch::TensorOptions().dtype(kernel.dtype())
                .device(kernel.device());
        torch::Tensor outMatrix = torch::zeros(numEntries, opts);

        dispatchKBuilding(grid, transform, kernel, indexMap, outMatrix);
        return {outMatrix};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& transform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        Variable kernel = ctx->saved_data["kernel"].toTensor();
        Variable indexMap = ctx->saved_data["indexMap"].toTensor();

        // Prepare grad input
        Variable gradOutMatrix = grad_output.at(0);

        // Prepare output
        torch::Tensor gradKernel = torch::zeros_like(kernel);

        dispatchKBuildingBackward(grid, transform, kernel, indexMap, gradOutMatrix, gradKernel);
        return {torch::Tensor(), torch::Tensor(), gradKernel, torch::Tensor(), torch::Tensor()};
    }
};

template <typename BufferType>
void dispatchRhsEvaluation(const nanovdb::GridHandle<BufferType>& grid,
                           const fvdb::VoxelCoordTransform& transform,
                           const torch::Tensor& pts,
                           const torch::Tensor& ptsKernel,
                           const torch::Tensor& gridKernel,
                           const torch::Tensor& gradKernelPts,
                           const torch::Tensor& ptsData,
                           torch::Tensor& outRhs,
                           unsigned nThreadsX = 128);

template <typename BufferType>
void dispatchRhsEvaluationBackward(const nanovdb::GridHandle<BufferType>& grid,
                                   const fvdb::VoxelCoordTransform& transform,
                                   const torch::Tensor& pts,
                                   const torch::Tensor& ptsKernel,
                                   const torch::Tensor& gridKernel,
                                   const torch::Tensor& gradKernelPts,
                                   const torch::Tensor& ptsData,
                                   const torch::Tensor& gradOutRhs,
                                   torch::Tensor& gradPtsKernel,
                                   torch::Tensor& gradGridKernel,
                                   torch::Tensor& gradGradKernelPts,
                                   torch::Tensor& gradPtsData,
                                   unsigned nThreadsX = 128);


template <typename BufferType>
struct RhsEvaluation : public torch::autograd::Function<RhsEvaluation<BufferType>> {
    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const fvdb::VoxelCoordTransform& transform,
                                 Variable pts, Variable ptsKernel,
                                 Variable gridKernel, Variable gradKernelPts,
                                 Variable ptsData) {
        // Save for backward
        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;
        ctx->saved_data["pts"] = pts;
        ctx->saved_data["ptsKernel"] = ptsKernel;
        ctx->saved_data["gridKernel"] = gridKernel;
        ctx->saved_data["gradKernelPts"] = gradKernelPts;
        ctx->saved_data["ptsData"] = ptsData;

        // Prepare output
        auto opts = torch::TensorOptions().dtype(pts.dtype())
                .device(pts.device());
        torch::Tensor outRhs = torch::zeros(gridKernel.size(0), opts);

        dispatchRhsEvaluation(grid, transform, pts, ptsKernel, gridKernel, gradKernelPts, ptsData, outRhs);
        return {outRhs};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& transform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        Variable pts = ctx->saved_data["pts"].toTensor();
        Variable ptsKernel = ctx->saved_data["ptsKernel"].toTensor();
        Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
        Variable gradKernelPts = ctx->saved_data["gradKernelPts"].toTensor();
        Variable ptsData = ctx->saved_data["ptsData"].toTensor();

        // Prepare grad input
        Variable gradOutRhs = grad_output.at(0);

        // Prepare output
        torch::Tensor gradPtsKernel = torch::zeros_like(ptsKernel);
        torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
        torch::Tensor gradGradKernelPts = torch::zeros_like(gradKernelPts);
        torch::Tensor gradPtsData = torch::zeros_like(ptsData);

        dispatchRhsEvaluationBackward<BufferType>(
                grid, transform, pts, ptsKernel, gridKernel, gradKernelPts, ptsData, gradOutRhs,
                gradPtsKernel, gradGridKernel, gradGradKernelPts, gradPtsData);
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
                gradPtsKernel, gradGridKernel, gradGradKernelPts, gradPtsData};
    }
};

template <typename BufferType>
void dispatchQgBuilding(const nanovdb::GridHandle<BufferType>& grid,
                        const fvdb::VoxelCoordTransform& transform,
                        const torch::Tensor& ptsPos,
                        const torch::Tensor& ptsKernel,
                        const torch::Tensor& gridKernel,
                        const torch::Tensor& gradKernelPts,
                        torch::Tensor& outIndexer,
                        torch::Tensor& outMatrix,
                        unsigned nThreadsX = 128);

template <typename BufferType>
void dispatchQgBuildingBackward(const nanovdb::GridHandle<BufferType>& grid,
                                const fvdb::VoxelCoordTransform& transform,
                                const torch::Tensor& ptsPos,
                                const torch::Tensor& ptsKernel,
                                const torch::Tensor& gridKernel,
                                const torch::Tensor& gradKernelPts,
                                const torch::Tensor& gradOutMatrix,
                                torch::Tensor& gradPtsKernel,
                                torch::Tensor& gradGridKernel,
                                torch::Tensor& gradGradKernelPts,
                                unsigned nThreadsX = 128);

template <typename BufferType>
struct QgBuilding : public torch::autograd::Function<QgBuilding<BufferType>> {
    static variable_list forward(AutogradContext *ctx,
                                 const nanovdb::GridHandle<BufferType>& grid,
                                 const fvdb::VoxelCoordTransform& transform,
                                 Variable pts, Variable ptsKernel,
                                 Variable gridKernel, Variable gradKernelPts, bool grad) {
        // Save for backward
        auto gridPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        gridPtr->ShareExternal((void*) &grid, caffe2::TypeMeta());
        ctx->saved_data["grid"] = gridPtr;
        auto transformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        transformPtr->ShareExternal((void*) &transform, caffe2::TypeMeta());
        ctx->saved_data["transform"] = transformPtr;
        ctx->saved_data["pts"] = pts;
        ctx->saved_data["ptsKernel"] = ptsKernel;
        ctx->saved_data["gridKernel"] = gridKernel;
        ctx->saved_data["gradKernelPts"] = gradKernelPts;

        // Prepare output
        auto opts = torch::TensorOptions().device(pts.device());
        torch::Tensor indexer = torch::full({pts.size(0), 27}, -1, opts.dtype(torch::kInt32));

        torch::Tensor qg;
        if (grad) {
            qg = torch::empty({pts.size(0), 27, 3}, opts.dtype(pts.dtype()));
        } else {
            qg = torch::empty({pts.size(0), 27}, opts.dtype(pts.dtype()));
        }

        ctx->mark_non_differentiable({indexer});

        dispatchQgBuilding(grid, transform, pts, ptsKernel, gridKernel, gradKernelPts, indexer, qg);
        return {qg, indexer};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const nanovdb::GridHandle<BufferType>& grid = *((nanovdb::GridHandle<BufferType>*) ctx->saved_data["grid"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& transform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["transform"].toBlob()->GetRaw());
        Variable pts = ctx->saved_data["pts"].toTensor();
        Variable ptsKernel = ctx->saved_data["ptsKernel"].toTensor();
        Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
        Variable gradKernelPts = ctx->saved_data["gradKernelPts"].toTensor();

        // Prepare grad input
        Variable gradOutQg = grad_output.at(0);

        // Prepare output
        torch::Tensor gradPtsKernel = torch::zeros_like(ptsKernel);
        torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
        torch::Tensor gradGradKernelPts = torch::zeros_like(gradKernelPts);

        dispatchQgBuildingBackward(grid, transform, pts, ptsKernel, gridKernel, gradKernelPts,
                                   gradOutQg, gradPtsKernel, gradGridKernel, gradGradKernelPts);

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
                gradPtsKernel, gradGridKernel, gradGradKernelPts, torch::Tensor()};
    }
};

void csrMatrixMultiplicationCUDA(
        const fvdb::VoxelCoordTransform& transformI,
        const fvdb::VoxelCoordTransform& transformJ,
        const torch::Tensor& coordsI, const torch::Tensor& coordsJ,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        torch::Tensor& outMatrix,
        unsigned nThreadsX = 16, unsigned nThreadsY = 10);

void csrMatrixMultiplicationBackwardCUDA(
        const fvdb::VoxelCoordTransform& transformI,
        const fvdb::VoxelCoordTransform& transformJ,
        const torch::Tensor& coordsI, const torch::Tensor& coordsJ,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        const torch::Tensor& gradOutMatrix,
        torch::Tensor& gradIValue,
        torch::Tensor& gradJValue,
        unsigned nThreadsX = 16,
        unsigned nThreadsY = 10);


struct CsrMatrixMultiplication : public torch::autograd::Function<CsrMatrixMultiplication> {
    static variable_list forward(AutogradContext *ctx,
                                 const fvdb::VoxelCoordTransform &iTransform,
                                 const fvdb::VoxelCoordTransform &jTransform,
                                 Variable coordsI, Variable coordsJ,
                                 Variable iValue, Variable jValue,
                                 Variable iRowPtr, Variable jRowPtr,
                                 Variable iColInds, Variable jColInds,
                                 Variable indexMap, int64_t numEntries) {
        // Save for backward
        auto iTransformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        iTransformPtr->ShareExternal((void *) &iTransform, caffe2::TypeMeta());
        ctx->saved_data["iTransform"] = iTransformPtr;
        auto jTransformPtr = c10::intrusive_ptr<caffe2::Blob>::make();
        jTransformPtr->ShareExternal((void *) &jTransform, caffe2::TypeMeta());
        ctx->saved_data["jTransform"] = jTransformPtr;
        ctx->saved_data["coordsI"] = coordsI;
        ctx->saved_data["coordsJ"] = coordsJ;
        ctx->saved_data["iValue"] = iValue;
        ctx->saved_data["jValue"] = jValue;
        ctx->saved_data["iRowPtr"] = iRowPtr;
        ctx->saved_data["jRowPtr"] = jRowPtr;
        ctx->saved_data["iColInds"] = iColInds;
        ctx->saved_data["jColInds"] = jColInds;
        ctx->saved_data["indexMap"] = indexMap;

        // Prepare output
        auto opts = torch::TensorOptions().dtype(iValue.dtype()).device(iValue.device());
        torch::Tensor outMatrix = torch::zeros(numEntries, opts);

        if (iValue.is_cuda()) {
            csrMatrixMultiplicationCUDA(iTransform, jTransform, coordsI, coordsJ, iValue, jValue,
                                        iRowPtr, jRowPtr, iColInds, jColInds, indexMap, outMatrix);
        } else {
            throw std::runtime_error("CPU version of CSR is not supported!");
        }

        return {outMatrix};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        const fvdb::VoxelCoordTransform& iTransform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["iTransform"].toBlob()->GetRaw());
        const fvdb::VoxelCoordTransform& jTransform = *((fvdb::VoxelCoordTransform*) ctx->saved_data["jTransform"].toBlob()->GetRaw());

        Variable coordsI = ctx->saved_data["coordsI"].toTensor();
        Variable coordsJ = ctx->saved_data["coordsJ"].toTensor();
        Variable iValue = ctx->saved_data["iValue"].toTensor();
        Variable jValue = ctx->saved_data["jValue"].toTensor();
        Variable iRowPtr = ctx->saved_data["iRowPtr"].toTensor();
        Variable jRowPtr = ctx->saved_data["jRowPtr"].toTensor();
        Variable iColInds = ctx->saved_data["iColInds"].toTensor();
        Variable jColInds = ctx->saved_data["jColInds"].toTensor();
        Variable indexMap = ctx->saved_data["indexMap"].toTensor();

        // Prepare grad input
        Variable gradOutMatrix = grad_output.at(0);

        // Prepare output
        torch::Tensor gradIValue = torch::zeros_like(iValue);
        torch::Tensor gradJValue = torch::zeros_like(jValue);

        if (iValue.is_cuda()) {
            csrMatrixMultiplicationBackwardCUDA(iTransform, jTransform, coordsI, coordsJ,
                                                iValue, jValue, iRowPtr, jRowPtr, iColInds, jColInds,
                                                indexMap, gradOutMatrix, gradIValue, gradJValue);
        } else {
            throw std::runtime_error("CPU version of CSR is not supported!");
        }

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                gradIValue, gradJValue,
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};
